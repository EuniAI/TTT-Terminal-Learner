"""Inference-only, geometrically growing tensor storage (no autograd graphs)."""
import torch


def append_cached(state, name, value, *, dim, limit):
    if torch.is_grad_enabled():
        raise RuntimeError('mutable KV storage is inference-only')
    old = state.get(name)
    length = 0 if old is None else old.shape[dim]
    needed = length + value.shape[dim]
    if needed > limit:
        raise ValueError('cache exceeds model context capacity')
    storage = state.get(name + '_storage')
    if storage is None or storage.shape[dim] < needed:
        capacity = min(limit, max(needed, 16 if storage is None else storage.shape[dim] * 2))
        shape = list(value.shape)
        shape[dim] = capacity
        storage = value.new_empty(shape)
        if length:
            storage.narrow(dim, 0, length).copy_(old)
    storage.narrow(dim, length, value.shape[dim]).copy_(value)
    result = storage.narrow(dim, 0, needed)
    state[name + '_storage'], state[name] = storage, result
    return result
