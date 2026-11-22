import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        scale = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return self.weight * (x * scale)


def build_rope_cache(head_dim, max_seq_len, theta, device=None):
    inverse = 1.0 / (theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32, device=device) / head_dim))
    positions = torch.arange(max_seq_len, dtype=torch.float32, device=device)
    angles = torch.outer(positions, inverse)
    return torch.cos(angles), torch.sin(angles)


def apply_rope(x, cos, sin):
    length = x.shape[-2]
    cos = cos[:length].view(1, 1, length, -1)
    sin = sin[:length].view(1, 1, length, -1)
    even, odd = x[..., 0::2], x[..., 1::2]
    rotated_even = even * cos - odd * sin
    rotated_odd = even * sin + odd * cos
    return torch.stack((rotated_even, rotated_odd), dim=-1).flatten(-2)


class SelfAttention(nn.Module):
    def __init__(self, d_model, n_head, dropout):
        super().__init__()
        self.n_head = n_head
        self.head_dim = d_model // n_head
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.proj = nn.Linear(d_model, d_model, bias=False)
        self.dropout = dropout

    def forward(self, x, cos, sin, mask):
        batch, length, _ = x.shape
        qkv = self.qkv(x).view(batch, length, 3, self.n_head, self.head_dim)
        query, key, value = qkv.permute(2, 0, 3, 1, 4)
        query = apply_rope(query, cos, sin)
        key = apply_rope(key, cos, sin)
        attended = F.scaled_dot_product_attention(
            query, key, value, attn_mask=mask, dropout_p=self.dropout if self.training else 0.0
        )
        merged = attended.transpose(1, 2).reshape(batch, length, -1)
        return self.proj(merged)


class SwiGLU(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.gate = nn.Linear(d_model, d_ff, bias=False)
        self.up = nn.Linear(d_model, d_ff, bias=False)
        self.down = nn.Linear(d_ff, d_model, bias=False)

    def forward(self, x):
        return self.down(F.silu(self.gate(x)) * self.up(x))


class Block(nn.Module):
    def __init__(self, d_model, n_head, d_ff, dropout):
        super().__init__()
        self.attention_norm = RMSNorm(d_model)
        self.attention = SelfAttention(d_model, n_head, dropout)
        self.feedforward_norm = RMSNorm(d_model)
        self.feedforward = SwiGLU(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, cos, sin, mask):
        x = x + self.dropout(self.attention(self.attention_norm(x), cos, sin, mask))
        x = x + self.dropout(self.feedforward(self.feedforward_norm(x)))
        return x


class ScreenerModel(nn.Module):
    def __init__(self, vocab_size, n_layer, n_head, d_model, d_ff, max_seq_len, dropout, rope_theta, n_labels, pad_id=0):
        super().__init__()
        if d_model % n_head:
            raise ValueError("d_model must divide evenly by n_head")
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.pad_id = pad_id
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.dropout = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([Block(d_model, n_head, d_ff, dropout) for _ in range(n_layer)])
        self.final_norm = RMSNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.lm_head.weight = self.embedding.weight
        self.classifier = nn.Linear(d_model, n_labels)
        cos, sin = build_rope_cache(d_model // n_head, max_seq_len, rope_theta)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)
        self.apply(self._init_weights)
        scale = 0.02 / math.sqrt(2 * n_layer)
        for block in self.blocks:
            nn.init.normal_(block.attention.proj.weight, mean=0.0, std=scale)
            nn.init.normal_(block.feedforward.down.weight, mean=0.0, std=scale)

    @staticmethod
    def _init_weights(module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def parameter_count(self, trainable_only=True):
        return sum(p.numel() for p in self.parameters() if p.requires_grad or not trainable_only)

    def _mask(self, ids, attention_mask):
        length = ids.shape[1]
        causal = torch.tril(torch.ones(length, length, dtype=torch.bool, device=ids.device))
        mask = causal.view(1, 1, length, length)
        if attention_mask is not None:
            keep = attention_mask.bool().view(ids.shape[0], 1, 1, length)
            mask = mask & keep
            diagonal = torch.eye(length, dtype=torch.bool, device=ids.device).view(1, 1, length, length)
            mask = mask | diagonal
        return mask

    def trunk(self, ids, attention_mask=None):
        if ids.shape[1] > self.max_seq_len:
            raise ValueError("sequence longer than max_seq_len")
        mask = self._mask(ids, attention_mask)
        x = self.dropout(self.embedding(ids))
        cos = self.rope_cos.to(x.dtype)
        sin = self.rope_sin.to(x.dtype)
        for block in self.blocks:
            x = block(x, cos, sin, mask)
        return self.final_norm(x)

    def forward_lm(self, ids, attention_mask=None):
        return self.lm_head(self.trunk(ids, attention_mask))

    def forward_cls(self, ids, attention_mask):
        hidden = self.trunk(ids, attention_mask)
        lengths = attention_mask.sum(dim=1).clamp(min=1) - 1
        pooled = hidden[torch.arange(hidden.shape[0], device=hidden.device), lengths]
        return self.classifier(pooled)
