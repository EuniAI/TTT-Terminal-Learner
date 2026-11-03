from __future__ import annotations

import math
import torch
from torch import nn
from torch.nn import functional as F
from terminal_learner.config import ModelConfig
from .attention import Attention
from .normalization import RMSNorm
from .backends import resolve_backend
from .placement import place_model

# Prime-MLP architecture follows EuniAI/TTT-Terminal-World.

class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        h, m = config.hidden_size, config.intermediate_size
        self.gate_proj, self.up_proj, self.down_proj = nn.Linear(h, m, False), nn.Linear(h, m, False), nn.Linear(m, h, False)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


def prime_mlp(x, weights, prefix):
    gate = F.linear(x, weights[prefix + "gate_proj.weight"])
    up = F.linear(x, weights[prefix + "up_proj.weight"])
    return F.linear(F.silu(gate) * up, weights[prefix + "down_proj.weight"])


class Block(nn.Module):
    def __init__(self, config, adaptive):
        super().__init__()
        self.input_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.self_attn = Attention(config)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.mlp = MLP(config)
        self.prime_norm = RMSNorm(config.hidden_size, config.rms_norm_eps) if adaptive else None

    def forward(self, x, positions, fast=None, index=None, kv_cache=None):
        device = self.input_layernorm.weight.device
        x, positions = x.to(device), positions.to(device)
        x = x + self.self_attn(self.input_layernorm(x), positions, kv_cache)
        if self.prime_norm is not None and fast is not None:
            x = x + prime_mlp(self.prime_norm(x), fast, f"layers.{index}.prime.")
        return x + self.mlp(self.post_attention_layernorm(x))


class TerminalTransformer(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        config.validate()
        self.config = config
        self.split = config.num_layers - config.adaptive_layers
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList(Block(config, i >= self.split) for i in range(config.num_layers))
        self.norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, False)
        if config.tie_embeddings:
            self.lm_head.weight = self.embed_tokens.weight
        self.to(dtype=config.torch_dtype)
        self.eval()

    def configure_backend(self):
        for layer in self.layers:
            layer.self_attn.flash_fn = resolve_backend(self.config, next(layer.parameters()).device)

    def fast_shapes(self):
        c = self.config
        return {f"layers.{i}.prime.{name}.weight": shape
                for i in range(self.split, c.num_layers)
                for name, shape in (("gate_proj", (c.intermediate_size, c.hidden_size)),
                                    ("up_proj", (c.intermediate_size, c.hidden_size)),
                                    ("down_proj", (c.hidden_size, c.intermediate_size)))}

    def fast_device(self, name):
        return next(self.layers[int(name.split('.')[1])].parameters()).device

    def place_fast(self, fast):
        return {k: v.detach().to(self.fast_device(k)).clone().requires_grad_(True)
                for k, v in fast.items()}

    def output_logits(self, x):
        return self.lm_head(self.norm(x.to(self.norm.weight.device)))

    def initialize_fast_weights(self):
        """Only called by the fresh-model factory; resume never invokes this."""
        reference = next(self.parameters())
        weights = {}
        for name, shape in self.fast_shapes().items():
            value = torch.empty(shape, device=self.fast_device(name), dtype=reference.dtype)
            if "down_proj" in name:
                value.zero_()
            else:
                nn.init.kaiming_uniform_(value, a=math.sqrt(5))
            weights[name] = value.requires_grad_(True)
        return weights

    def validate_fast(self, fast):
        expected = self.fast_shapes()
        if set(fast) != set(expected):
            raise ValueError("fast-weight names mismatch; initialization fallback is forbidden")
        reference = next(self.parameters())
        storages = {(p.device, p.untyped_storage().data_ptr()) for p in self.parameters()}
        for name, value in fast.items():
            if tuple(value.shape) != expected[name] or value.device != self.fast_device(name) or value.dtype != reference.dtype:
                raise ValueError(f"fast-weight shape/device/dtype mismatch: {name}")
            pointer = (value.device, value.untyped_storage().data_ptr())
            if pointer in storages:
                raise ValueError("theta/phi or phi/phi storage alias")
            storages.add(pointer)

    def prefix(self, ids, positions):
        if ids.shape[1] > self.config.max_seq_len:
            raise ValueError("input exceeds model capacity")
        x = self.embed_tokens(ids.to(self.embed_tokens.weight.device))
        for i in range(self.split):
            x = self.layers[i](x, positions)
        return x

    def suffix(self, x, positions, fast, *, logits_start=0):
        for i in range(self.split, len(self.layers)):
            x = self.layers[i](x, positions, fast, i)
        return self.output_logits(x[:, logits_start:])

    def forward(self, ids, fast, positions=None, *, logits_start=0):
        if positions is None:
            positions = torch.arange(ids.shape[1], device=ids.device).expand_as(ids)
        return self.suffix(self.prefix(ids, positions), positions, fast, logits_start=logits_start)

    @torch.no_grad()
    def suffix_cached(self, hidden, fast, cache=None):
        """Generate from shared frozen-prefix states with event-local suffix KV."""
        if cache is None:
            cache = [{} for _ in self.layers[self.split:]]
        start = cache[0]['k'].shape[-2] if cache and 'k' in cache[0] else 0
        x = hidden[:, start:]
        positions = torch.arange(start, hidden.shape[1], device=hidden.device).unsqueeze(0)
        for i, entry in zip(range(self.split, len(self.layers)), cache):
            x = self.layers[i](x, positions, fast, i, kv_cache=entry)
        return self.output_logits(x[:, -1:]), cache

    @torch.no_grad()
    def forward_cached(self, ids, fast, cache=None):
        """Prefill or extend an event-local cache; return next-token logits.

        The caller must discard this cache when fast weights or model weights
        change, and at event boundaries. Nothing is stored on the model.
        """
        if cache is None:
            cache = [{} for _ in self.layers]
        if len(cache) != len(self.layers):
            raise ValueError("KV cache layer count mismatch")
        lengths = [entry["k"].shape[-2] if "k" in entry else 0 for entry in cache]
        if len(set(lengths)) != 1:
            raise ValueError("KV cache lengths mismatch")
        start = lengths[0]
        if ids.shape[1] < 1 or start + ids.shape[1] > self.config.max_seq_len:
            raise ValueError("empty input or input exceeds model capacity")
        positions = torch.arange(start, start + ids.shape[1], device=ids.device).expand_as(ids)
        x = self.embed_tokens(ids.to(self.embed_tokens.weight.device))
        for i, layer in enumerate(self.layers):
            x = layer(x, positions, fast, i, kv_cache=cache[i])
        # Generation only needs the final position, especially with large vocabs.
        return self.output_logits(x[:, -1:]), cache


def create_model(config, device="cpu", *, devices=None, layer_counts=None):
    resolve_backend(config, devices[0] if devices else device)  # Fail before allocating model weights.
    model = place_model(TerminalTransformer(config), device, devices=devices, layer_counts=layer_counts)
    model.configure_backend()
    fast = model.initialize_fast_weights()
    model.validate_fast(fast)
    return model, fast
