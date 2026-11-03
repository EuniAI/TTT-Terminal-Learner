"""Contiguous layer placement, without wrapping or breaking higher-order autograd."""
from __future__ import annotations

import torch


def validate_devices(devices):
    if not devices:
        raise ValueError('model_parallel_devices must not be empty')
    result = [torch.device(d) for d in devices]
    if len(set(result)) != len(result):
        raise ValueError('model parallel devices must be distinct')
    for device in result:
        if device.type != 'cuda' or device.index is None:
            raise ValueError('model parallel devices require explicit CUDA indices, e.g. cuda:0')
        if device.index >= torch.cuda.device_count():
            raise ValueError(f'model parallel device unavailable: {device}')
    return result


def layer_partition(model, stages, counts=None):
    """Minimize the maximum estimated static training bytes per contiguous stage.

    18 bytes/slow parameter: BF16 weight, FP32 master/Adam moments/gradient.
    Fast-weight allowance is 12 bytes/parameter. Activations and unroll graphs
    are excluded: this is a placement heuristic, NOT a peak-memory guarantee.
    """
    n = len(model.layers)
    if not 1 <= stages <= n:
        raise ValueError('number of stages must be between 1 and number of layers')
    if counts is not None:
        if len(counts) != stages or any(type(x) is not int or x < 1 for x in counts) or sum(counts) != n:
            raise ValueError('layer counts must be positive, match devices, and sum to num_layers')
        return list(counts)
    size = lambda module: sum(p.numel() for p in module.parameters()) * 18
    costs = [size(layer) + (3 * model.config.hidden_size * model.config.intermediate_size * 12
                           if i >= model.split else 0) for i, layer in enumerate(model.layers)]
    first = size(model.embed_tokens)
    last = size(model.norm) + (0 if model.config.tie_embeddings else size(model.lm_head))
    if model.config.tie_embeddings:
        first += last
        last = 0
    prefix = [0]
    for cost in costs:
        prefix.append(prefix[-1] + cost)
    best = {(0, 0): (0, [])}
    for k in range(1, stages + 1):
        for end in range(k, n + 1):
            candidates = []
            for start in range(k - 1, end):
                if (k - 1, start) in best:
                    previous, partition = best[k - 1, start]
                    cost = prefix[end] - prefix[start] + (first if k == 1 else 0) + (last if k == stages else 0)
                    candidates.append((max(previous, cost), partition + [end - start]))
            best[k, end] = min(candidates, key=lambda item: item[0])
    return best[stages, n][1]


def place_model(model, device='cpu', *, devices=None, layer_counts=None):
    if devices is None:
        if layer_counts is not None:
            raise ValueError('layer counts require model parallel devices')
        return model.to(device)
    devices = validate_devices(devices)
    counts = layer_partition(model, len(devices), layer_counts)
    model.embed_tokens.to(devices[0])
    offset = 0
    for device, count in zip(devices, counts):
        for layer in model.layers[offset:offset + count]:
            layer.to(device)
        offset += count
    output_device = devices[0] if model.config.tie_embeddings else devices[-1]
    model.norm.to(output_device)
    model.lm_head.to(output_device)
    model.layer_placement = dict(devices=[str(d) for d in devices], layer_counts=counts)
    return model


def cuda_devices(model):
    return sorted({p.device for p in model.parameters() if p.is_cuda}, key=lambda d: d.index)
