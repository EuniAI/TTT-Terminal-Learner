from __future__ import annotations

from contextlib import nullcontext
import torch
from torch import nn
from torch.nn import functional as F
from .normalization import RMSNorm
from .cache import append_cached

def causal_attention(q, k, v, backend="math", flash_fn=None):
    if backend == "flashbob":
        if flash_fn is None:
            raise RuntimeError("FlashBoB backend has not passed startup validation")
        with torch.cuda.device(q.device) if q.is_cuda else nullcontext():
            return flash_fn(q, k, v, is_causal=True, scale=q.shape[-1] ** -0.5, window_size=0)
    # Explicit attention keeps the mixed second derivatives needed by meta-training.
    if backend != "math":
        raise ValueError("unknown attention backend")
    ratio = q.shape[1] // k.shape[1]
    if ratio != 1:
        k, v = k.repeat_interleave(ratio, 1), v.repeat_interleave(ratio, 1)
    dtype = torch.float64 if q.dtype == torch.float64 else torch.float32
    scores = q.to(dtype) @ k.to(dtype).transpose(-2, -1) * q.shape[-1] ** -0.5
    mask = torch.ones(q.shape[-2], k.shape[-2], device=q.device, dtype=torch.bool).triu(1)
    return (scores.masked_fill(mask, -torch.inf).softmax(-1) @ v.to(dtype)).to(q.dtype)


class Attention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        h, d = config.hidden_size, config.hidden_size // config.num_heads
        self.q_proj = nn.Linear(h, config.num_heads * d, bias=config.qkv_bias)
        self.k_proj = nn.Linear(h, config.num_kv_heads * d, bias=config.qkv_bias)
        self.v_proj = nn.Linear(h, config.num_kv_heads * d, bias=config.qkv_bias)
        self.o_proj = nn.Linear(h, h, False)
        self.q_norm = RMSNorm(d, config.rms_norm_eps) if config.qk_norm else None
        self.k_norm = RMSNorm(d, config.rms_norm_eps) if config.qk_norm else None
        self.flash_fn = None

    def forward(self, x, positions, kv_cache=None):
        c = self.config
        b, n, _ = x.shape
        d = c.hidden_size // c.num_heads
        q = self.q_proj(x).view(b, n, c.num_heads, d).transpose(1, 2)
        k = self.k_proj(x).view(b, n, c.num_kv_heads, d).transpose(1, 2)
        v = self.v_proj(x).view(b, n, c.num_kv_heads, d).transpose(1, 2)
        if self.q_norm is not None:
            q, k = self.q_norm(q), self.k_norm(k)
        work_dtype = torch.float64 if x.dtype == torch.float64 else torch.float32
        inv = 1 / c.rope_theta ** (torch.arange(0, d, 2, device=x.device, dtype=work_dtype) / d)
        angles = positions.to(work_dtype)[..., None] * inv
        angles = torch.cat((angles, angles), -1).unsqueeze(1)
        cos, sin = angles.cos().to(x.dtype), angles.sin().to(x.dtype)
        def rotate(t):
            return torch.cat((-t[..., d // 2:], t[..., :d // 2]), -1)
        q, k = q * cos + rotate(q) * sin, k * cos + rotate(k) * sin
        past_length = 0
        if kv_cache is not None:
            if "k" in kv_cache:
                past_length = kv_cache["k"].shape[-2]
            k = append_cached(kv_cache, 'k', k, dim=2, limit=c.max_seq_len)
            v = append_cached(kv_cache, 'v', v, dim=2, limit=c.max_seq_len)
        alignment_attention = getattr(self, "_alignment_attention", None)
        if past_length:
            # Cached queries are at the END of the key sequence. A square/top-left
            # causal mask would incorrectly hide the cached context. Single-token
            # decoding may attend to every key; chunks need an offset causal mask.
            ratio = c.num_heads // c.num_kv_heads
            keys = k.repeat_interleave(ratio, 1) if ratio != 1 else k
            values = v.repeat_interleave(ratio, 1) if ratio != 1 else v
            mask = None
            if n > 1:
                mask = (torch.arange(k.shape[-2], device=x.device)[None, :]
                        <= past_length + torch.arange(n, device=x.device)[:, None])
            # Inference only: native SDPA supports rectangular attention; leave
            # the differentiable math attention training path unchanged.
            out = F.scaled_dot_product_attention(q, keys, values, attn_mask=mask)
        else:
            out = (alignment_attention(q, k, v) if alignment_attention is not None
                   else causal_attention(q, k, v, c.attention_backend, self.flash_fn))
        return self.o_proj(out.transpose(1, 2).reshape(b, n, c.hidden_size))
