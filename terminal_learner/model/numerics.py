"""Validate nested checkpoint/optimizer tensors before accepting their state."""
import torch


def validate_finite_tree(value, label):
    if isinstance(value, torch.Tensor):
        if (value.is_floating_point() or value.is_complex()) and not bool(torch.isfinite(value).all()):
            raise FloatingPointError(f'nonfinite {label}')
    elif isinstance(value, dict):
        for key, child in value.items():
            validate_finite_tree(child, f'{label}.{key}')
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            validate_finite_tree(child, f'{label}[{index}]')
