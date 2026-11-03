from __future__ import annotations

import math
import torch
from torch.nn import functional as F

def detach_fast(fast):
    return {k: v.detach().requires_grad_(True) for k, v in fast.items()}


def copy_fast(fast, device=None):
    return {k: v.detach().to(device=device or v.device).clone().requires_grad_(True)
            for k, v in fast.items()}


def finite(tensors, label):
    if any(not bool(torch.isfinite(t).all()) for t in tensors):
        raise FloatingPointError(f'nonfinite {label}')


def inner_update(fast, loss, *, lr, create_graph, clip_norm=1.0, return_stats=False):
    if not math.isfinite(lr) or lr <= 0 or not math.isfinite(clip_norm) or clip_norm <= 0:
        raise ValueError('inner learning rate and clip norm must be finite and positive')
    grads = torch.autograd.grad(loss, tuple(fast.values()), create_graph=create_graph)
    finite(grads, 'inner gradients')
    work = [g if g.dtype == torch.float64 else g.float() for g in grads]
    norm = (sum(g.square().sum().to(work[0].device) for g in work) + 1e-30).sqrt()
    factor = (clip_norm / norm).clamp(max=1)
    updated = {k: v - lr * (g * factor.to(g.device)).to(v.dtype)
               for (k, v), g in zip(fast.items(), grads)}
    finite(updated.values(), 'fast weights')
    if return_stats:
        with torch.no_grad():
            update_norm = sum((updated[k].detach().to(norm.dtype) - v.detach().to(norm.dtype)).square().sum().to(norm.device)
                              for k, v in fast.items()).sqrt()
        return updated, {'inner_grad_norm': float(norm.detach()),
                         'inner_update_norm': float(update_norm),  # Actual stored parameter change.
                         'inner_planned_update_norm': float((norm * factor * lr).detach()),
                         'inner_clipped': bool(norm.detach() > clip_norm)}
    return updated


def token_nll(model, fast, encoder, prompt, targets):
    """One forward on complete text, with no silent context/target truncation."""
    if not targets or not prompt:
        raise ValueError('empty prompt or target')
    sequence = list(prompt) + list(targets)
    if len(sequence) > model.config.max_seq_len:
        raise ValueError(f'complete prompt and targets require {len(sequence)} tokens, '
                         f'exceeding model context capacity {model.config.max_seq_len}; '
                         'no text was truncated')
    device = next(model.parameters()).device
    inputs = torch.tensor([sequence[:-1]], device=device)
    logits = model(inputs, fast, logits_start=len(prompt) - 1)[0]
    work = logits if logits.dtype == torch.float64 else logits.float()
    total = F.cross_entropy(work, torch.tensor(targets, device=logits.device), reduction='sum')
    finite([total], 'token loss')
    return total, len(targets)


def action_loss(model, fast, encoder, task, history, action, *, reasoning_content=None):
    prompt, targets = encoder.action_example(task, history, action, reasoning_content=reasoning_content)
    return token_nll(model, fast, encoder, prompt, targets)


def observation_loss(model, fast, encoder, task, history, action, output, *, reasoning_content=None):
    """Token-weighted mean from one forward over the retained observation.

    The caller performs exactly one inner update after this loss is assembled.
    """
    prompt, targets = encoder.observation_example(task, history, action, output, reasoning_content=reasoning_content)
    nll, count = token_nll(model, fast, encoder, prompt, targets)
    return nll / count, count, False

def window_loss(model, fast, encoder, episode, turns, history, config,
                *, differentiable=True):
    """Unroll obs-based SGD and supervise subsequent actions through its graph.

    differentiable=False is ONLY for the detached ablation in tests.
    There is no theta optimizer step inside an inner update.
    """
    losses, count, stats = [], 0, []
    for index, turn in enumerate(turns):
        nll, n = action_loss(model, fast, encoder, episode.task, history, turn.action,
                             reasoning_content=turn.reasoning_content)
        losses.append(nll)
        count += n
        stat = {'step_id': turn.step_id, 'action_nll': float(nll.detach()) / n,
                'action_tokens': n, 'observation_tokens': 0,
                'observation_truncated': False, 'missing_observation': turn.observation is None}
        if turn.observation is not None:
            obs, tokens, truncated = observation_loss(
                model, fast, encoder, episode.task, history, turn.action, turn.observation,
                reasoning_content=turn.reasoning_content)
            fast, inner_stats = inner_update(fast, obs, lr=config.inner_lr,
                                clip_norm=config.inner_clip,
                                create_graph=differentiable and index < len(turns) - 1,
                                return_stats=True)
            stat.update(inner_stats)
            if not differentiable:
                fast = detach_fast(fast)
            history = encoder.history(history, turn.action, turn.observation, reasoning_content=turn.reasoning_content)
            stat.update(observation_nll=float(obs.detach()), observation_tokens=tokens,
                        observation_truncated=truncated)
        stats.append(stat)
    if not count:
        raise ValueError('empty training window')
    return torch.stack(losses).sum() / count, fast, history, stats
