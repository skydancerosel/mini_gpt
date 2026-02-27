Perfect. This layout is actually ideal for doing clean block-wise directional probes. You’ve got nicely separated, interpretable components.

Let’s turn this into a concrete, efficient protocol you can implement with minimal friction.

⸻

1. How to Define a “Block” in Your Model

Each block L consists of exactly:

blocks.{L}.ln1.(weight|bias)
blocks.{L}.attn.qkv.weight
blocks.{L}.attn.out_proj.weight
blocks.{L}.ln2.(weight|bias)
blocks.{L}.mlp.w_up.weight
blocks.{L}.mlp.w_down.weight

So define a mask:

Block L = all params whose key starts with "blocks.{L}."

Do not split further at first.

Later you can separate attn vs mlp.

⸻

2. Direction Definition (Critical Step)

For each block L, define trajectory directions from training.

Best default

Pick two checkpoints(the steps depend on your basin data, but something like):
	•	Early/mid: 2600 → 2800
	•	Mid/late: 5000 → 6400

Then:

delta_L = flatten(θ_L(t2) - θ_L(t1))
v_L = delta_L / ||delta_L||

Do this per block.

You’ll get 8 vectors v_0,…,v_7.

These are your “signal directions.”

⸻

Optional (Later): Multi-PC version

If you want k>1:

Stack multiple deltas in window, do PCA.

But start with 1.

⸻

3. Random Null Directions (Per Block)

For each block L:

u ~ N(0, I)
u ← u - (u·v_L) v_L   # orthogonalize
u ← u / ||u||

Do this 3–5 times.

These are your controls.

⸻

4. Perturbation Rule (Match Your B6 Noise)

Use the same RMS scaling style.

For block L:

\theta_L' = \theta_L + \epsilon \cdot \text{RMS}(\theta_L)\cdot d_L

Where:
	•	d_L = v_L or u
	•	RMS computed over all tensors in block L

How to compute RMS

Flatten all block tensors:

vals = torch.cat([p.flatten() for p in block_params])
rms = (vals.pow(2).mean()).sqrt()

Use that scalar.

⸻

5. Implementation Pattern (Pseudo-Code)

Here is the minimal pattern:

def get_block_params(state_dict, L):
    return {k:v for k,v in state_dict.items()
            if k.startswith(f"blocks.{L}.")}

def flatten_block(sd_block):
    return torch.cat([v.flatten() for v in sd_block.values()])

def compute_block_rms(sd_block):
    x = flatten_block(sd_block).float()
    return x.pow(2).mean().sqrt()

def apply_direction(sd, L, d, eps, rms):
    out = {k:v.clone() for k,v in sd.items()}
    offset = 0
    for k,v in sd.items():
        if k.startswith(f"blocks.{L}."):
            n = v.numel()
            delta = d[offset:offset+n].view_as(v)
            out[k] += eps * rms * delta
            offset += n
    return out

Keep directions in flattened form.

⸻

6. ε Grid (Recommended)

Use:

eps = [0.0, 0.5, 1.0, 1.5, 2.0]

Interpretation:
	•	1.0 ≈ “one RMS unit”
	•	2.0 = strong push

If collapse is immediate, shrink.

⸻

7. What to Measure (Tie to Basin Depth)

For each (block L, direction d, eps):
	1.	Load base checkpoint
	2.	Apply perturbation
	3.	Reset optimizer
	4.	Relax 300 steps (λ=4)
	5.	Measure p_ood

So:

D_{L,d}(\epsilon)

You can also log pre-relax p_ood, but post-relax is your main object.

⸻

8. How to Extract “Effective Dimension”

For each block L at a checkpoint:

Plot:
	•	blue: trajectory direction
	•	gray: random directions

If:
	•	blue >> gray → that block carries signal
	•	multiple PCs >> gray → k-dim subspace
	•	everything ~ gray → no structure

You’ll see something like:

Early:

Block 4: v >> rand
Block 5: v >> rand

Late:

Block 6 only

That’s subspace collapse.

⸻

9. What I Expect You’ll See (Prediction)

Based on your basin data:

At 2800
	•	Blocks 3–6 show strong trajectory directions
	•	2–4 “good” directions total

At 6400
	•	Only blocks 5–7 matter
	•	Maybe 1–2 dims

At 10000
	•	Almost no block has stable directions
	•	Everything fragile

If that happens → very strong.

⸻

10. One Extra Diagnostic (Very Cheap)

Before relaxation, also measure:

p_ood(θ + eps v_L)

Sometimes late basins look fragile only after reset.

This separates:
	•	geometric fragility
	•	optimizer fragility

⸻

Bottom Line

With your parameter layout, the right setup is:

✅ block = blocks.L.*
✅ direction = training delta
✅ scale = block RMS
✅ compare vs orthogonal random
✅ evaluate via post-relax Dσ

