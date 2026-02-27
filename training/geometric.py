"""
Geometric monitoring signals for weight-space dynamics.

1. Commutator defect: measures non-commutativity of gradient steps
   (signals loss-landscape curvature changes that precede capability emergence).
2. PCA on weight updates: tracks dimensionality of the update trajectory.
"""

import numpy as np
import torch
import torch.nn.functional as F


# ═══════════════════════════════════════════════════════════════════════════
# Parameter utilities
# ═══════════════════════════════════════════════════════════════════════════

def flatten_params(model):
    """Flatten all model parameters into a single 1-D tensor (detached, CPU)."""
    return torch.cat([p.detach().cpu().reshape(-1) for p in model.parameters()])


def write_params(model, theta):
    """Write a flat parameter vector back into the model."""
    with torch.no_grad():
        offset = 0
        for p in model.parameters():
            n = p.numel()
            p.copy_(theta[offset:offset + n].view_as(p))
            offset += n


def flatten_grad(model):
    """Flatten current gradients into a single 1-D tensor."""
    return torch.cat([
        (p.grad if p.grad is not None else torch.zeros_like(p)).flatten()
        for p in model.parameters()
    ])


# ═══════════════════════════════════════════════════════════════════════════
# Commutator defect
# ═══════════════════════════════════════════════════════════════════════════

def commutator_defect(model, loss_fn_A, loss_fn_B, device, eta=1e-3):
    """
    Compute scale-normalized commutator defect.

    Given two loss functions (batches), measure how much the order of
    gradient steps matters: ||theta_AB - theta_BA|| / (||step_A|| * ||step_B||).

    Args:
        model: the neural network
        loss_fn_A: callable() -> scalar loss (on batch A)
        loss_fn_B: callable() -> scalar loss (on batch B)
        device: torch device
        eta: step size for the two gradient steps

    Returns:
        defect: scalar, the commutator defect
        delta: 1-D tensor, the commutator vector (theta_AB - theta_BA)
    """
    was_training = model.training
    model.train()

    def get_grad(loss_fn):
        model.zero_grad(set_to_none=True)
        loss = loss_fn()
        loss.backward()
        return flatten_grad(model).to(device)

    theta0 = flatten_params(model).to(device)
    gA = get_grad(loss_fn_A)
    gB = get_grad(loss_fn_B)

    # Path A then B
    write_params(model, theta0 - eta * gA)
    gB1 = get_grad(loss_fn_B)
    theta_AB = theta0 - eta * gA - eta * gB1

    # Path B then A
    write_params(model, theta0 - eta * gB)
    gA1 = get_grad(loss_fn_A)
    theta_BA = theta0 - eta * gB - eta * gA1

    # Restore original parameters
    write_params(model, theta0)
    if not was_training:
        model.eval()

    delta = theta_AB - theta_BA
    norm_A = (eta * gA).norm()
    norm_B = (eta * gB).norm()
    eps = 1e-12

    defect = (delta.norm() / (norm_A * norm_B + eps)).item()
    return defect, delta.detach().cpu()


def commutator_defect_from_dataloader(model, dataloader, device, eta=1e-3, K=5):
    """
    Compute median commutator defect over K random batch pairs.

    Args:
        model: the neural network
        dataloader: provides batches of (input_ids, targets, probe_mask)
        device: torch device
        eta: step size
        K: number of commutator samples

    Returns:
        dict with median defect, individual defects, and median delta vector
    """
    data_iter = iter(dataloader)
    defects = []
    deltas = []

    for _ in range(K):
        # Get two different batches
        try:
            batch_A = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            batch_A = next(data_iter)
        try:
            batch_B = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            batch_B = next(data_iter)

        input_A, target_A, _ = batch_A
        input_B, target_B, _ = batch_B
        input_A, target_A = input_A.to(device), target_A.to(device)
        input_B, target_B = input_B.to(device), target_B.to(device)

        def loss_A():
            _, loss = model(input_A, target_A)
            return loss

        def loss_B():
            _, loss = model(input_B, target_B)
            return loss

        d, delta = commutator_defect(model, loss_A, loss_B, device, eta=eta)
        defects.append(d)
        deltas.append(delta)

    # Find median
    med_idx = np.argsort(defects)[len(defects) // 2]
    return {
        "defect_median": float(np.median(defects)),
        "defect_p25": float(np.percentile(defects, 25)),
        "defect_p75": float(np.percentile(defects, 75)),
        "defects": defects,
        "median_delta": deltas[med_idx],
    }


# ═══════════════════════════════════════════════════════════════════════════
# PCA on weight update trajectory
# ═══════════════════════════════════════════════════════════════════════════

class UpdatePCA:
    """
    Maintains PCA over successive weight updates (Δθ = θ_{t+1} - θ_t).
    Logs explained variance of top PCs and effective dimensionality.
    """

    def __init__(self, n_components=10):
        self.n_components = n_components
        self.updates = []  # list of flattened Δθ vectors
        self.prev_params = None

    def record(self, model):
        """Record current parameters. Computes Δθ if we have a previous snapshot."""
        current = flatten_params(model)
        if self.prev_params is not None:
            delta = current - self.prev_params
            if delta.norm() > 1e-15:
                self.updates.append(delta)
        self.prev_params = current.clone()

    def compute_pca(self):
        """
        Compute PCA over accumulated updates.

        Returns:
            dict with explained_variance_ratio for top PCs,
            k_star_95 and k_star_99 (number of PCs for 95%/99% variance).
        """
        if len(self.updates) < 2:
            return None

        # Stack updates into matrix [n_updates, n_params]
        X = torch.stack(self.updates).float()
        # Center
        X = X - X.mean(dim=0, keepdim=True)

        # SVD (on the smaller dimension)
        n, p = X.shape
        if n < p:
            # Compute on n x n covariance
            cov = X @ X.T / (n - 1)
            eigenvalues, _ = torch.linalg.eigh(cov)
            eigenvalues = eigenvalues.flip(0)  # descending
        else:
            _, S, _ = torch.linalg.svd(X, full_matrices=False)
            eigenvalues = (S ** 2) / (n - 1)

        total_var = eigenvalues.sum().item()
        if total_var < 1e-15:
            return None

        explained = eigenvalues / total_var
        explained = explained[:self.n_components].tolist()

        # Pad if fewer components
        while len(explained) < self.n_components:
            explained.append(0.0)

        # k* for 95% and 99%
        cumsum = torch.cumsum(eigenvalues / total_var, dim=0)
        k_95 = (cumsum < 0.95).sum().item() + 1
        k_99 = (cumsum < 0.99).sum().item() + 1

        return {
            "explained_variance_ratio": explained,
            "k_star_95": k_95,
            "k_star_99": k_99,
            "n_updates": len(self.updates),
            "total_variance": total_var,
        }


# ═══════════════════════════════════════════════════════════════════════════
# Emergence detection
# ═══════════════════════════════════════════════════════════════════════════

def detect_emergence(accuracy_trace, threshold=0.8, consecutive=3):
    """
    Detect capability emergence time.

    Args:
        accuracy_trace: list of (step, accuracy) tuples
        threshold: accuracy threshold
        consecutive: number of consecutive evals above threshold

    Returns:
        emergence_step or None
    """
    count = 0
    for step, acc in accuracy_trace:
        if acc >= threshold:
            count += 1
            if count >= consecutive:
                # Return the step where it first crossed
                return accuracy_trace[accuracy_trace.index((step, acc)) - consecutive + 1][0]
        else:
            count = 0
    return None


def detect_defect_onset(defect_trace, baseline_window=10, sigma_mult=3.0,
                        sustained=2):
    """
    Detect defect onset time (first sustained elevation above baseline).

    Args:
        defect_trace: list of (step, defect_value) tuples
        baseline_window: number of initial measurements for baseline stats
        sigma_mult: number of sigmas above baseline mean
        sustained: number of consecutive elevated measurements

    Returns:
        onset_step or None
    """
    if len(defect_trace) < baseline_window + sustained:
        return None

    baseline_vals = [d for _, d in defect_trace[:baseline_window]]
    baseline_mean = np.mean(baseline_vals)
    baseline_std = np.std(baseline_vals) + 1e-15
    threshold = baseline_mean + sigma_mult * baseline_std

    count = 0
    for step, val in defect_trace[baseline_window:]:
        if val > threshold:
            count += 1
            if count >= sustained:
                # Return the step where elevation started
                idx = defect_trace.index((step, val)) - sustained + 1
                return defect_trace[idx][0]
        else:
            count = 0
    return None
