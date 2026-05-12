"""Build a natural-text induction batch from OpenWebText.

Each example is 256 tokens with an embedded induction pattern:
  position i:    token T
  position i+1:  token S
  ... (filler) ...
  position k:    token T again (LATEST occurrence in the document)

We use this to measure attention from position k to canonical target
positions per example:
  induction:  i+1  (S, the induction target)
  duplicate:  i    (first T)
  prev-token: k-1
  self:       k
  first:      0
  local:      mean over [k-5, k-4, k-3, k-2]
  baseline:   50 random positions avoiding all of the above

Mirrors the synthetic `build_induction_batch` from
analyses/induction_heads_per_head_124m.py but uses natural-text contexts
to avoid the synthetic-batch BOS attractor artifact.

Saves a .pt file with tokens, query_pos, first_T_pos, targets.
"""
from __future__ import annotations

import argparse
import time

import torch
from datasets import load_dataset
from transformers import AutoTokenizer


def build(n_examples: int = 2000, seq_len: int = 256, seed: int = 42,
          min_T_id: int = 100, min_filler: int = 20,
          min_first_T_pos: int = 0, max_docs: int = 200000,
          tokenizer_name: str = "EleutherAI/gpt-neox-20b") -> dict:
    """Returns a dict of stacked tensors and metadata.

    min_first_T_pos: if >0, only accept examples where first_T_pos >= this value.
        Set to 20 to push the first-T (and the induction target at first_T+1)
        well outside the BOS-attractor zone, removing the BOS/induction-target
        position-overlap confound.
    """
    tok = AutoTokenizer.from_pretrained(tokenizer_name)
    ds = load_dataset("Skylion007/openwebtext", split="train", streaming=True)

    tokens_out, qpos_out, fT_out, targets_out = [], [], [], []
    n_docs_tried = 0
    t0 = time.time()
    skip_reasons = {"short": 0, "no_pattern": 0, "T_too_common": 0}

    for doc in ds:
        n_docs_tried += 1
        if n_docs_tried > max_docs:
            print(f"  reached max_docs={max_docs}, stopping with {len(tokens_out)} examples")
            break
        ids = tok(doc["text"], truncation=True, max_length=seq_len,
                  return_tensors="pt").input_ids[0]
        if len(ids) < seq_len:
            skip_reasons["short"] += 1
            continue

        # Scan: find pairs (T = ids[i], S = ids[i+1]) where T reappears
        # at some k > i+1 with at least `min_filler` tokens between
        # first-T and second-T. Restrict i >= min_first_T_pos to keep
        # the induction target out of the BOS-attractor zone.
        # Take the latest valid k to maximize context.
        found = False
        for i in range(min_first_T_pos, seq_len - 2 - min_filler):
            T = ids[i].item()
            S = ids[i + 1].item()
            if T < min_T_id or S < min_T_id:
                continue
            if T == S:  # degenerate; skip
                continue
            tail = ids[i + 1 + min_filler:]
            matches = (tail == T).nonzero(as_tuple=True)[0]
            if len(matches) > 0:
                k = (i + 1 + min_filler + matches[-1]).item()
                if k >= seq_len:
                    continue
                tokens_out.append(ids.clone())
                qpos_out.append(k)
                fT_out.append(i)
                targets_out.append(S)
                found = True
                break  # one example per doc
        if not found:
            skip_reasons["no_pattern"] += 1

        if len(tokens_out) >= n_examples:
            break
        if n_docs_tried % 500 == 0:
            print(f"  tried {n_docs_tried} docs, kept {len(tokens_out)}/{n_examples}  "
                  f"[{time.time()-t0:.0f}s]", flush=True)

    print(f"\nKept {len(tokens_out)}/{n_examples} after trying {n_docs_tried} docs.")
    print(f"Skip reasons: {skip_reasons}")

    tokens = torch.stack(tokens_out)
    qpos = torch.tensor(qpos_out, dtype=torch.long)
    fT = torch.tensor(fT_out, dtype=torch.long)
    targets = torch.tensor(targets_out, dtype=torch.long)

    # Quick sanity stats
    gaps = qpos - fT
    print(f"\nGap (query - first_T) stats: "
          f"min={gaps.min().item()}, median={int(gaps.median().item())}, "
          f"max={gaps.max().item()}, mean={gaps.float().mean():.1f}")
    print(f"Query position stats: min={qpos.min().item()}, median={int(qpos.median().item())}, "
          f"max={qpos.max().item()}")

    return {
        "tokens": tokens,
        "query_pos": qpos,
        "first_T_pos": fT,
        "targets": targets,
        "seq_len": seq_len,
        "n_examples": len(tokens),
        "tokenizer": tokenizer_name,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-examples", type=int, default=2000)
    ap.add_argument("--seq-len", type=int, default=256)
    ap.add_argument("--min-T-id", type=int, default=100)
    ap.add_argument("--min-filler", type=int, default=20)
    ap.add_argument("--min-first-T-pos", type=int, default=0,
                    help="filter examples to require first_T_pos >= this; "
                         "set to 20 to push induction target out of BOS-attractor zone")
    ap.add_argument("--max-docs", type=int, default=200000,
                    help="max OpenWebText docs to scan before stopping")
    ap.add_argument("--out", default="natural_induction_batch.pt")
    args = ap.parse_args()

    data = build(n_examples=args.n_examples, seq_len=args.seq_len,
                 min_T_id=args.min_T_id, min_filler=args.min_filler,
                 min_first_T_pos=args.min_first_T_pos, max_docs=args.max_docs)
    torch.save(data, args.out)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
