"""
TinyStories dataset with long-range key-retrieval probe injection.

- Base: TinyStories (standard train split)
- Probe: Insert KEY sentence early and QUERY late in a fraction p_probe of examples
- Codewords: uppercase strings chosen to be single-token under the tokenizer
"""

import random
from typing import List, Tuple, Optional

import torch
from torch.utils.data import Dataset, DataLoader


# ═══════════════════════════════════════════════════════════════════════════
# Codeword generation: find uppercase strings that are single tokens
# ═══════════════════════════════════════════════════════════════════════════

def find_single_token_codewords(tokenizer, n_codewords=512, seed=42):
    """Find uppercase strings that tokenize to exactly one token.

    Strategy: check uppercase words from a candidate pool.
    """
    rng = random.Random(seed)

    # Candidate pool: all uppercase tokens in the vocab
    candidates = []
    for token_str in tokenizer.get_vocab():
        # Strip leading space/Ġ that GPT2 tokenizers use
        clean = token_str.lstrip("Ġ ").strip()
        if clean.isupper() and clean.isalpha() and len(clean) >= 4:
            # Verify it's truly single-token
            encoded = tokenizer.encode(clean, add_special_tokens=False)
            if len(encoded) == 1:
                candidates.append(clean)

    # If not enough single-token uppercase words, generate synthetic ones
    if len(candidates) < n_codewords:
        # Also try with leading space (common in GPT2 tokenizer)
        for token_str in tokenizer.get_vocab():
            clean = token_str.lstrip("Ġ ").strip()
            if clean.isupper() and clean.isalpha() and len(clean) >= 3:
                encoded = tokenizer.encode(" " + clean, add_special_tokens=False)
                if len(encoded) == 1:
                    if clean not in candidates:
                        candidates.append(clean)

    # Also try common uppercase words
    extra_words = [
        "ALPHA", "BETA", "GAMMA", "DELTA", "OMEGA", "SIGMA", "THETA",
        "ZERO", "HERO", "KING", "QUEEN", "STAR", "MOON", "SUN",
        "FIRE", "WATER", "EARTH", "WIND", "STORM", "LIGHT", "DARK",
        "RED", "BLUE", "GREEN", "GOLD", "SILVER", "BLACK", "WHITE",
        "NORTH", "SOUTH", "EAST", "WEST", "DAWN", "DUSK", "NOON",
        "APPLE", "BERRY", "CANDY", "DOVE", "EAGLE", "FROST", "GHOST",
        "HAWK", "IVORY", "JADE", "KELP", "LEMON", "MAPLE", "NIGHT",
        "OLIVE", "PEARL", "QUARTZ", "ROSE", "STONE", "TIGER", "UNITY",
        "VIOLET", "WHALE", "XENON", "YARN", "ZEBRA",
    ]
    for w in extra_words:
        if w not in candidates:
            encoded = tokenizer.encode(w, add_special_tokens=False)
            if len(encoded) <= 2:  # allow up to 2 tokens
                candidates.append(w)

    # Sort before shuffle: tokenizer.get_vocab() iteration order depends on
    # Python's hash randomization, so candidates list is non-deterministic.
    # Sorting ensures reproducible codewords across processes.
    candidates.sort()
    rng.shuffle(candidates)
    codewords = candidates[:n_codewords]

    if len(codewords) < n_codewords:
        print(f"  WARNING: Only found {len(codewords)} codewords "
              f"(requested {n_codewords}). Using what we have.")

    return codewords


# ═══════════════════════════════════════════════════════════════════════════
# Probe templates
# ═══════════════════════════════════════════════════════════════════════════

KEY_TEMPLATES = [
    "The secret code is {code}.",
    "Remember the word {code}.",
    "The special word is {code}.",
    "The password is {code}.",
    "The magic word is {code}.",
]

QUERY_TEMPLATES = [
    "What is the secret code? {code}",
    "What was the word? {code}",
    "What is the special word? {code}",
    "What is the password? {code}",
    "The magic word was {code}",
]


def make_probe_pair(codeword, rng):
    """Create a key-query template pair for a given codeword."""
    idx = rng.randint(0, len(KEY_TEMPLATES) - 1)
    key_str = KEY_TEMPLATES[idx].format(code=codeword)
    query_str = QUERY_TEMPLATES[idx].format(code=codeword)
    return key_str, query_str


# ═══════════════════════════════════════════════════════════════════════════
# Dataset
# ═══════════════════════════════════════════════════════════════════════════

class TinyStoriesProbeDataset(Dataset):
    """
    TinyStories dataset with optional probe injection.

    For each example:
    - With probability p_probe, inject a KEY sentence early and QUERY late.
    - The rest is standard language modeling (predict next token).
    - For probe examples, we also track where the codeword appears in the
      query so we can evaluate probe accuracy.
    """

    def __init__(self, texts: List[str], tokenizer, seq_len: int,
                 codewords: List[str], p_probe: float = 0.05,
                 gap_range: Tuple[int, int] = (5, 30),
                 seed: int = 42):
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.codewords = codewords
        self.p_probe = p_probe
        self.gap_range = gap_range
        self.rng = random.Random(seed)

        # Store raw texts; tokenize on-the-fly in __getitem__
        self.texts = [t for t in texts if len(t) >= 40]
        self._cache = {}  # idx -> token list (lazy tokenization cache)

    def _get_tokens(self, idx):
        """Lazy tokenization with caching."""
        if idx not in self._cache:
            self._cache[idx] = self.tokenizer.encode(
                self.texts[idx], add_special_tokens=False
            )
        return self._cache[idx]

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        tokens = self._get_tokens(idx)
        is_probe = self.rng.random() < self.p_probe and len(self.codewords) > 0

        if is_probe:
            return self._make_probe_example(tokens)
        else:
            return self._make_lm_example(tokens)

    def _make_lm_example(self, tokens):
        """Standard LM example: truncate/pad to seq_len+1, return input/target."""
        if len(tokens) > self.seq_len + 1:
            start = self.rng.randint(0, len(tokens) - self.seq_len - 1)
            tokens = tokens[start:start + self.seq_len + 1]
        elif len(tokens) < self.seq_len + 1:
            pad_id = self.tokenizer.eos_token_id or 0
            tokens = tokens + [pad_id] * (self.seq_len + 1 - len(tokens))

        input_ids = torch.tensor(tokens[:self.seq_len], dtype=torch.long)
        targets = torch.tensor(tokens[1:self.seq_len + 1], dtype=torch.long)
        probe_mask = torch.zeros(self.seq_len, dtype=torch.bool)
        return input_ids, targets, probe_mask

    def _make_probe_example(self, tokens):
        """Inject key-query probe into the token sequence."""
        codeword = self.rng.choice(self.codewords)
        key_str, query_str = make_probe_pair(codeword, self.rng)

        key_tokens = self.tokenizer.encode(key_str, add_special_tokens=False)
        query_tokens = self.tokenizer.encode(query_str, add_special_tokens=False)
        code_tokens = self.tokenizer.encode(codeword, add_special_tokens=False)

        # We need: [prefix] [key] [middle] [query] [suffix]
        # gap = distance from end of key to start of query (in tokens)
        gap_min, gap_max = self.gap_range
        desired_gap = self.rng.randint(gap_min, gap_max)

        total_injected = len(key_tokens) + desired_gap + len(query_tokens)
        budget = self.seq_len + 1 - total_injected

        if budget < 10:
            # Not enough room, fall back to LM example
            return self._make_lm_example(tokens)

        # Split budget between prefix and suffix
        prefix_len = self.rng.randint(2, max(3, budget // 3))
        prefix_len = min(prefix_len, len(tokens))

        # Build the injected sequence
        prefix = tokens[:prefix_len]

        # Middle tokens from the original text
        middle_start = min(prefix_len, len(tokens) - 1)
        middle_tokens = tokens[middle_start:middle_start + desired_gap]
        if len(middle_tokens) < desired_gap:
            # Pad middle if not enough original tokens
            pad_id = self.tokenizer.eos_token_id or 0
            middle_tokens = middle_tokens + [pad_id] * (desired_gap - len(middle_tokens))

        full_seq = prefix + key_tokens + middle_tokens + query_tokens

        # Truncate or pad to seq_len + 1
        if len(full_seq) > self.seq_len + 1:
            full_seq = full_seq[:self.seq_len + 1]
        elif len(full_seq) < self.seq_len + 1:
            # Add more from original text as suffix
            remaining = self.seq_len + 1 - len(full_seq)
            suffix_start = min(prefix_len + desired_gap, len(tokens))
            suffix = tokens[suffix_start:suffix_start + remaining]
            if len(suffix) < remaining:
                pad_id = self.tokenizer.eos_token_id or 0
                suffix = suffix + [pad_id] * (remaining - len(suffix))
            full_seq = full_seq + suffix

        full_seq = full_seq[:self.seq_len + 1]

        input_ids = torch.tensor(full_seq[:self.seq_len], dtype=torch.long)
        targets = torch.tensor(full_seq[1:self.seq_len + 1], dtype=torch.long)

        # Mark positions where the codeword appears in the query answer
        # The probe_mask marks target positions corresponding to codeword tokens
        probe_mask = torch.zeros(self.seq_len, dtype=torch.bool)
        query_start_in_full = len(prefix) + len(key_tokens) + len(middle_tokens)
        # Find where code_tokens start within query_tokens
        query_str_no_code = query_str.replace(codeword, "").strip()
        pre_code_tokens = self.tokenizer.encode(
            query_str_no_code.split(codeword)[0] if codeword in query_str else "",
            add_special_tokens=False
        )
        code_start = query_start_in_full + len(query_tokens) - len(code_tokens)
        # Mark positions (shifted by -1 because targets are shifted)
        for i in range(len(code_tokens)):
            pos = code_start + i - 1  # -1 for target shift
            if 0 <= pos < self.seq_len:
                probe_mask[pos] = True

        return input_ids, targets, probe_mask


# ═══════════════════════════════════════════════════════════════════════════
# Probe-only evaluation dataset (fixed)
# ═══════════════════════════════════════════════════════════════════════════

class ProbeEvalDataset(Dataset):
    """Fixed evaluation set of probe-only examples."""

    def __init__(self, tokenizer, seq_len, codewords, n_examples,
                 gap_range, seed=42):
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.examples = []

        rng = random.Random(seed)
        pad_id = tokenizer.eos_token_id or 0

        for _ in range(n_examples):
            codeword = rng.choice(codewords)
            key_str, query_str = make_probe_pair(codeword, rng)
            code_tokens = tokenizer.encode(codeword, add_special_tokens=False)
            key_tokens = tokenizer.encode(key_str, add_special_tokens=False)
            query_tokens = tokenizer.encode(query_str, add_special_tokens=False)

            gap = rng.randint(gap_range[0], gap_range[1])

            # Build: [filler] [key] [filler gap] [query] [filler]
            total_injected = len(key_tokens) + gap + len(query_tokens)
            prefix_len = rng.randint(2, 10)
            # Use random common tokens as filler
            filler_tokens = [rng.randint(100, 5000) for _ in range(seq_len + 1)]

            full_seq = (filler_tokens[:prefix_len] + key_tokens +
                       filler_tokens[prefix_len:prefix_len + gap] +
                       query_tokens)

            # Pad/truncate
            if len(full_seq) < seq_len + 1:
                full_seq = full_seq + [pad_id] * (seq_len + 1 - len(full_seq))
            full_seq = full_seq[:seq_len + 1]

            input_ids = torch.tensor(full_seq[:seq_len], dtype=torch.long)
            targets = torch.tensor(full_seq[1:seq_len + 1], dtype=torch.long)

            # Mark codeword positions in target
            probe_mask = torch.zeros(seq_len, dtype=torch.bool)
            code_start = prefix_len + len(key_tokens) + gap + len(query_tokens) - len(code_tokens)
            for i in range(len(code_tokens)):
                pos = code_start + i - 1
                if 0 <= pos < seq_len:
                    probe_mask[pos] = True

            self.examples.append({
                "input_ids": input_ids,
                "targets": targets,
                "probe_mask": probe_mask,
                "codeword": codeword,
                "code_tokens": code_tokens,
                "code_start": code_start,
            })

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        ex = self.examples[idx]
        return ex["input_ids"], ex["targets"], ex["probe_mask"]


# ═══════════════════════════════════════════════════════════════════════════
# Data loading helpers
# ═══════════════════════════════════════════════════════════════════════════

def load_tinystories(tokenizer_name="roneneldan/TinyStories", split="train",
                     max_examples=None):
    """Load TinyStories texts from HuggingFace datasets."""
    from datasets import load_dataset
    ds = load_dataset("roneneldan/TinyStories", split=split)
    texts = []
    for i, example in enumerate(ds):
        if max_examples is not None and i >= max_examples:
            break
        text = example.get("text", "")
        if text.strip():
            texts.append(text)
    return texts


def build_datasets(cfg, codewords_path=None):
    """Build all datasets needed for training and evaluation.

    Args:
        cfg: Config object
        codewords_path: Optional path to a JSON file containing a pre-built
            codewords list (key: "codewords"). If provided, load codewords
            from file instead of generating them. This ensures reproducibility
            across processes (tokenizer.get_vocab() iteration order can vary).
    """
    import json as _json
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(cfg.tokenizer_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if codewords_path is not None:
        print(f"Loading codewords from {codewords_path}...")
        with open(codewords_path) as f:
            codewords = _json.load(f)["codewords"]
        print(f"  Loaded {len(codewords)} codewords")
    else:
        print("Finding single-token codewords...")
        codewords = find_single_token_codewords(tokenizer, cfg.n_codewords)
        print(f"  Found {len(codewords)} codewords")

    max_train = getattr(cfg, 'max_train_examples', None)
    max_val = getattr(cfg, 'max_val_examples', None)

    print(f"Loading TinyStories train split (max={max_train})...")
    train_texts = load_tinystories(cfg.tokenizer_name, split="train",
                                   max_examples=max_train)
    print(f"  {len(train_texts)} training texts")

    print(f"Loading TinyStories validation split (max={max_val})...")
    val_texts = load_tinystories(cfg.tokenizer_name, split="validation",
                                 max_examples=max_val)
    print(f"  {len(val_texts)} validation texts")

    # Training dataset with probe injection
    train_dataset = TinyStoriesProbeDataset(
        texts=train_texts,
        tokenizer=tokenizer,
        seq_len=cfg.seq_len,
        codewords=codewords,
        p_probe=cfg.p_probe,
        gap_range=cfg.probe_gap_train,
        seed=cfg.seed,
    )

    # Validation dataset (no probe injection, just LM)
    val_dataset = TinyStoriesProbeDataset(
        texts=val_texts,
        tokenizer=tokenizer,
        seq_len=cfg.seq_len,
        codewords=codewords,
        p_probe=0.0,
        gap_range=cfg.probe_gap_train,
        seed=cfg.seed + 1,
    )

    # Fixed probe evaluation sets
    print("Building probe eval sets...")
    probe_eval_in = ProbeEvalDataset(
        tokenizer=tokenizer,
        seq_len=cfg.seq_len,
        codewords=codewords,
        n_examples=cfg.n_probe_eval_in,
        gap_range=cfg.probe_gap_train,
        seed=cfg.seed + 100,
    )
    probe_eval_ood = ProbeEvalDataset(
        tokenizer=tokenizer,
        seq_len=cfg.seq_len,
        codewords=codewords,
        n_examples=cfg.n_probe_eval_ood,
        gap_range=cfg.probe_gap_ood,
        seed=cfg.seed + 200,
    )

    print(f"  probe_eval_in:  {len(probe_eval_in)} examples, gaps {cfg.probe_gap_train}")
    print(f"  probe_eval_ood: {len(probe_eval_ood)} examples, gaps {cfg.probe_gap_ood}")

    return {
        "tokenizer": tokenizer,
        "codewords": codewords,
        "train_dataset": train_dataset,
        "val_dataset": val_dataset,
        "probe_eval_in": probe_eval_in,
        "probe_eval_ood": probe_eval_ood,
    }
