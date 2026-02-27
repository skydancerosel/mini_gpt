"""
Capability control via subspace suppression.

Periodically compute differential gradient direction (probe vs LM),
maintain a low-rank basis of probe-specific update directions, and
optionally project them out of the training update.
"""

import torch
import torch.nn.functional as F


class SubspaceSuppressor:
    """
    Maintains a differential subspace basis S and can suppress probe-specific
    update directions from the gradient.

    Every `control_every` steps:
    1. Compute g_lm = gradient on normal LM batch
    2. Compute g_probe = gradient on probe-only batch
    3. v = normalize(g_probe - g_lm)
    4. Maintain PCA over recent v's (top-k)
    5. On each training step: Δθ ← Δθ - λ * P_S(Δθ)
    """

    def __init__(self, model, control_k=3, lam=0.0, max_history=50):
        """
        Args:
            model: the neural network
            control_k: rank of the suppression subspace
            lam: suppression strength (0 = off, 1 = full projection out)
            max_history: max differential directions to keep for PCA
        """
        self.model = model
        self.control_k = control_k
        self.lam = lam
        self.max_history = max_history
        self.diff_directions = []  # list of normalized v vectors
        self.basis = None  # [P, k] orthonormal basis

    def update_basis(self, lm_batch, probe_batch, device):
        """
        Compute differential gradient direction and update the basis.

        Args:
            lm_batch: (input_ids, targets) for normal LM
            probe_batch: (input_ids, targets) for probe-only examples
            device: torch device
        """
        if self.lam == 0.0:
            return

        self.model.train()

        # g_lm
        self.model.zero_grad(set_to_none=True)
        lm_input, lm_target = lm_batch
        lm_input, lm_target = lm_input.to(device), lm_target.to(device)
        _, loss_lm = self.model(lm_input, lm_target)
        loss_lm.backward()
        g_lm = torch.cat([
            (p.grad.flatten() if p.grad is not None else torch.zeros(p.numel(), device=device))
            for p in self.model.parameters()
        ])

        # g_probe
        self.model.zero_grad(set_to_none=True)
        probe_input, probe_target = probe_batch
        probe_input, probe_target = probe_input.to(device), probe_target.to(device)
        _, loss_probe = self.model(probe_input, probe_target)
        loss_probe.backward()
        g_probe = torch.cat([
            (p.grad.flatten() if p.grad is not None else torch.zeros(p.numel(), device=device))
            for p in self.model.parameters()
        ])

        # Differential direction
        v = g_probe - g_lm
        v_norm = v.norm()
        if v_norm < 1e-15:
            return
        v = v / v_norm

        self.diff_directions.append(v.detach().cpu())
        if len(self.diff_directions) > self.max_history:
            self.diff_directions.pop(0)

        # Build basis via PCA over recent diff directions
        self._rebuild_basis()

    def _rebuild_basis(self):
        """Rebuild orthonormal basis from recent differential directions."""
        if len(self.diff_directions) < 1:
            self.basis = None
            return

        V = torch.stack(self.diff_directions)  # [n, P]
        k = min(self.control_k, len(self.diff_directions), V.shape[1])

        # SVD to get top-k directions
        try:
            U, S, Vh = torch.linalg.svd(V.float(), full_matrices=False)
            self.basis = Vh[:k].T  # [P, k]
            # Orthonormalize
            self.basis, _ = torch.linalg.qr(self.basis, mode="reduced")
        except Exception:
            self.basis = None

    def suppress_gradient(self):
        """
        Project out the probe-specific subspace from the current gradient.
        Call this after loss.backward() but before optimizer.step().

        Modifies gradients in-place: g ← g - λ * P_S(g)
        where P_S is projection onto the suppression subspace.
        """
        if self.lam == 0.0 or self.basis is None:
            return

        device = next(self.model.parameters()).device
        B = self.basis.to(device)  # [P, k]

        # Collect gradient
        grads = []
        params_with_grad = []
        for p in self.model.parameters():
            if p.requires_grad and p.grad is not None:
                grads.append(p.grad.flatten())
                params_with_grad.append(p)

        if not grads:
            return

        g = torch.cat(grads)

        # Only suppress on transformer blocks (exclude embeddings/head)
        # For simplicity, apply to all parameters but could be restricted
        g_proj = B @ (B.T @ g)
        g_new = g - self.lam * g_proj

        # Write back
        offset = 0
        for p in params_with_grad:
            n = p.grad.numel()
            p.grad.copy_(g_new[offset:offset + n].view_as(p.grad))
            offset += n
