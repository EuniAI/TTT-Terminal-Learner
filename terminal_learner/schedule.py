"""Outer learning rate indexed by completed windows, with a persisted horizon."""
import math


def make_schedule(config, windows_per_epoch, epochs):
    config.validate()
    if config.outer_lr_schedule == 'constant':
        return None
    total = windows_per_epoch * epochs
    if total < 3:
        raise ValueError('warmup_cosine requires at least 3 planned training windows')
    warmup = min(math.ceil(total * config.outer_warmup_ratio), total - 2)
    return {'total_windows': total, 'warmup_windows': warmup}


def outer_learning_rate(config, completed, plan):
    if config.outer_lr_schedule == 'constant':
        return config.outer_lr
    if not isinstance(plan, dict):
        raise ValueError('scheduled training requires a saved outer_lr_plan')
    total, warmup = plan.get('total_windows'), plan.get('warmup_windows')
    if (type(total) is not int or total < 3 or type(warmup) is not int
            or not 0 <= warmup <= total - 2):
        raise ValueError('invalid outer_lr_plan')
    if type(completed) is not int or completed < 0:
        raise ValueError('completed windows must be a nonnegative integer')
    if warmup and completed < warmup:
        return config.outer_lr * completed / warmup
    progress = min(1.0, (completed - warmup) / (total - 1 - warmup))
    return config.outer_end_lr + (config.outer_lr - config.outer_end_lr) * (
        1 + math.cos(math.pi * progress)) / 2
