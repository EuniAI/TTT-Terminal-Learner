from __future__ import annotations

import time
from dataclasses import dataclass
import torch
from .meta_model import copy_fast, detach_fast, finite, window_loss
from .optimizer import SlowWeightAdamW
from .placement import cuda_devices
from terminal_learner.schedule import outer_learning_rate


@dataclass
class WindowWork:
    gradients: tuple
    state: dict
    stats: list
    action_nll: float
    action_tokens: int
    started: float


class Trainer:
    """Local differentiable unroll; optional gradient reduction happens outside it."""
    def __init__(self, model, initial_fast, encoder, config, *, lr_plan=None):
        config.validate()
        outer_learning_rate(config, 0, lr_plan)
        self.lr_plan = lr_plan
        model.validate_fast(initial_fast)
        self.model, self.initial_fast, self.encoder, self.config = model, initial_fast, encoder, config
        model.requires_grad_(True)
        model.train()
        self.optimizer = SlowWeightAdamW(model.parameters(), lr=config.outer_lr,
                                          weight_decay=config.weight_decay)
        self.windows_completed = 0
        self.episode_state = None

    def compute_window(self, episode, *, state=None, limit_turns=None):
        """Compute local meta-gradients without changing theta or committing state."""
        if limit_turns is not None and limit_turns < 2:
            raise ValueError('limit_turns must be >= 2')
        turns = episode.turns[:limit_turns] if limit_turns is not None else episode.turns
        if state is None:
            state = dict(trial=episode.trial, next_turn=0, fast=copy_fast(self.initial_fast),
                         history=self.encoder.initial_history(episode.initial_observation),
                         total_nll=0.0, total_tokens=0, windows=0, truncated=0,
                         observation_nll_sum=0.0, observation_tokens=0)
        if state['trial'] != episode.trial or not 0 <= state['next_turn'] < len(turns):
            raise ValueError('invalid checkpoint episode cursor')
        device = next(self.model.parameters()).device
        fast = state['fast']
        if any(v.device != self.model.fast_device(k) or not v.requires_grad for k, v in fast.items()):
            fast = self.model.place_fast(fast)
        self.model.validate_fast(fast)
        for gpu in cuda_devices(self.model):
            torch.cuda.synchronize(gpu)
            torch.cuda.reset_peak_memory_stats(gpu)
        started = time.perf_counter()
        self.optimizer.zero_grad(set_to_none=True)
        start = state['next_turn']
        window = turns[start:start + self.config.unroll_steps]
        loss, fast, history, stats = window_loss(
            self.model, fast, self.encoder, episode, window, state['history'], self.config)
        gradients = torch.autograd.grad(loss, tuple(self.model.parameters()))
        finite(gradients, 'outer gradients')
        n = sum(s['action_tokens'] for s in stats)
        obs_tokens = sum(s['observation_tokens'] for s in stats)
        obs_sum = sum(s.get('observation_nll', 0.0) * s['observation_tokens'] for s in stats)
        updated = dict(trial=episode.trial, next_turn=start + len(window), fast=fast, history=history,
            total_nll=state['total_nll'] + float(loss.detach()) * n,
            total_tokens=state['total_tokens'] + n, windows=state['windows'] + 1,
            truncated=state['truncated'] + sum(s['observation_truncated'] for s in stats),
            observation_nll_sum=state.get('observation_nll_sum', 0.0) + obs_sum,
            observation_tokens=state.get('observation_tokens', 0) + obs_tokens)
        return WindowWork(tuple(g.detach() for g in gradients), updated, stats, float(loss.detach()), n, started)

    def finish_window(self, work, *, gradients=None):
        """Apply one outer update. Idle data-parallel ranks also participate here."""
        device = next(self.model.parameters()).device
        lr = outer_learning_rate(self.config, self.windows_completed, self.lr_plan)
        for group in self.optimizer.param_groups:
            group['lr'] = lr
        if gradients is None:
            if work is None:
                raise ValueError('idle ranks require reduced gradients')
            gradients = work.gradients
        finite(gradients, 'reduced outer gradients')
        self.optimizer.set_gradients(gradients)
        grad_norm = torch.linalg.vector_norm(torch.stack([
            torch.linalg.vector_norm(p.grad.detach()).to(device) for p in self.optimizer.master_parameters]))
        finite([grad_norm], 'outer gradient norm')
        self.optimizer.step()
        finite(tuple(self.model.parameters()), 'slow weights')
        self.windows_completed += 1
        stats = work.stats if work is not None else []
        if work is not None:
            self.episode_state = {**work.state, 'fast': detach_fast(work.state['fast'])}
        for gpu in cuda_devices(self.model):
            torch.cuda.synchronize(gpu)
        elapsed = time.perf_counter() - work.started if work is not None else 0.0
        n = work.action_tokens if work is not None else 0
        inner = [s for s in stats if 'inner_grad_norm' in s]
        obs_tokens = sum(s['observation_tokens'] for s in stats)
        obs_sum = sum(s.get('observation_nll', 0.0) * s['observation_tokens'] for s in stats)
        return {'trial': work.state['trial'] if work is not None else None,
                'window': self.windows_completed,
                'action_nll': work.action_nll if work is not None else None, 'action_tokens': n,
                'observation_tokens': obs_tokens,
                'observation_nll': obs_sum / obs_tokens if obs_tokens else None,
                'window_seconds': elapsed, 'target_tokens_per_second': (n + obs_tokens) / elapsed if elapsed else 0,
                'inner_grad_norm': sum(s['inner_grad_norm'] for s in inner) / max(1, len(inner)),
                'inner_update_norm': sum(s['inner_update_norm'] for s in inner) / max(1, len(inner)),
                'inner_planned_update_norm': sum(s['inner_planned_update_norm'] for s in inner) / max(1, len(inner)),
                'inner_clip_ratio': sum(s['inner_clipped'] for s in inner) / max(1, len(inner)),
                'peak_allocated_bytes': max((torch.cuda.max_memory_allocated(d) for d in cuda_devices(self.model)), default=None),
                'peak_reserved_bytes': max((torch.cuda.max_memory_reserved(d) for d in cuda_devices(self.model)), default=None),
                'device_peak_allocated_bytes': {str(d): torch.cuda.max_memory_allocated(d) for d in cuda_devices(self.model)},
                'outer_lr': lr, 'inner_lr': self.config.inner_lr,
                'outer_grad_norm': float(grad_norm), 'steps': stats}

    def train_episode(self, episode, *, limit_turns=None, on_window=None, state=None, should_stop=lambda: False):
        if self.config.action_supervision == 'success' and episode.reward != 1:
            return {'trial': episode.trial, 'skipped': 'action_supervision=success'}
        if limit_turns is not None and limit_turns < 2:
            raise ValueError('limit_turns must be >= 2')
        turns = episode.turns[:limit_turns] if limit_turns is not None else episode.turns
        if not turns:
            return {'trial': episode.trial, 'skipped': 'no recorded tool actions'}
        # Keep the episode state even if a stop is requested before its first window.
        if state is None:
            state = dict(trial=episode.trial, next_turn=0, fast=copy_fast(self.initial_fast),
                         history=self.encoder.initial_history(episode.initial_observation),
                         total_nll=0.0, total_tokens=0, windows=0, truncated=0,
                         observation_nll_sum=0.0, observation_tokens=0)
        self.episode_state = state
        if state['trial'] != episode.trial or not 0 <= state['next_turn'] <= len(turns):
            raise ValueError('invalid checkpoint episode cursor')
        while state['next_turn'] < len(turns) and not should_stop():
            work = self.compute_window(episode, state=state, limit_turns=limit_turns)
            row = self.finish_window(work)
            state = self.episode_state
            del work
            if on_window:
                on_window(row)
        return self.episode_report(episode)

    def episode_report(self, episode):
        state = self.episode_state
        return {'trial': episode.trial, 'windows': state['windows'],
                'action_nll': state['total_nll'] / state['total_tokens'] if state['total_tokens'] else None,
                'action_tokens': state['total_tokens'], 'observation_tokens': state.get('observation_tokens', 0),
                'observation_nll': state.get('observation_nll_sum', 0) / state['observation_tokens']
                                   if state.get('observation_tokens', 0) else None,
                'truncated_observations': state['truncated'], 'used_turns': state['next_turn'],
                'available_turns': len(episode.turns)}
