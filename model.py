"""
GPT-small model for TinyStories language modeling.

4-layer causal transformer with tied input/output embeddings.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalSelfAttention(nn.Module):
    def __init__(self, d_model, n_head, seq_len, dropout=0.0):
        super().__init__()
        assert d_model % n_head == 0
        self.d_model = d_model
        self.n_head = n_head
        self.head_dim = d_model // n_head

        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.attn_drop = nn.Dropout(dropout)
        self.resid_drop = nn.Dropout(dropout)

        # causal mask
        self.register_buffer(
            "bias",
            torch.tril(torch.ones(seq_len, seq_len)).view(1, 1, seq_len, seq_len),
        )

    def forward(self, x):
        B, T, C = x.shape
        qkv = self.qkv(x)  # [B, T, 3*C]
        q, k, v = qkv.split(self.d_model, dim=2)

        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.head_dim))
        att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)

        y = att @ v  # [B, nh, T, hd]
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_drop(self.out_proj(y))


class MLP(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.0):
        super().__init__()
        self.w_up = nn.Linear(d_model, d_ff, bias=False)
        self.w_down = nn.Linear(d_ff, d_model, bias=False)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        return self.drop(self.w_down(F.gelu(self.w_up(x))))


class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_head, d_ff, seq_len, dropout=0.0):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_head, seq_len, dropout)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = MLP(d_model, d_ff, dropout)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class GPTModel(nn.Module):
    """
    GPT-small for TinyStories with tied embeddings.
    Pre-norm architecture (LayerNorm before attention/MLP).
    """

    def __init__(self, vocab_size, seq_len, d_model, n_layer, n_head, d_ff,
                 dropout=0.0):
        super().__init__()
        self.seq_len = seq_len
        self.d_model = d_model
        self.n_layer = n_layer

        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(seq_len, d_model)
        self.drop = nn.Dropout(dropout)

        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_head, d_ff, seq_len, dropout)
            for _ in range(n_layer)
        ])
        self.ln_f = nn.LayerNorm(d_model)

        # Tied output embedding (no separate lm_head weight)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.lm_head.weight = self.tok_emb.weight  # weight tying

        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, idx, targets=None):
        """
        idx:     [B, T] token indices
        targets: [B, T] target token indices (optional)
        Returns: logits [B, T, V], loss (if targets given)
        """
        B, T = idx.shape
        assert T <= self.seq_len, f"Sequence length {T} > max {self.seq_len}"

        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)
        h = self.drop(self.tok_emb(idx) + self.pos_emb(pos))

        for block in self.blocks:
            h = block(h)

        h = self.ln_f(h)
        logits = self.lm_head(h)  # [B, T, V]

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=-100,
            )

        return logits, loss

    def count_params(self):
        # Don't double-count tied weights
        seen = set()
        total = 0
        for p in self.parameters():
            if id(p) not in seen:
                seen.add(id(p))
                total += p.numel()
        return total
