"""Persist and restore the unrounded slow weights through the public train API."""
import json
import os
import signal

import torch
import pytest
import importlib

from terminal_learner.config import EncodingConfig, ModelConfig, TrainConfig
from terminal_learner.dataloader.tokenizer import Tokenizer
from terminal_learner.model.transformer import create_model
from terminal_learner.train import train


@pytest.mark.parametrize("stop_at", ["epoch", "window", "window_end", "logging_failure", "SIGINT", "SIGTERM"])
@pytest.mark.parametrize("schedule", ["constant", "warmup_cosine"])
@pytest.mark.parametrize('overflow_tail', [False, True])
def test_bf16_qwen3_training_resume(tmp_path, monkeypatch, capsys, schedule, stop_at, overflow_tail):
    torch.set_num_threads(1)
    trajectory = {
        'agent': {'name': 'terminus-2', 'model_name': 'anthropic/claude-opus-4-6'},
        'steps': [{'source': 'user', 'message': 'Task Description:\nlist files\nCurrent terminal state:$'}]
    }
    for i in range(3):
        trajectory['steps'].append({
            'source': 'agent', 'step_id': i + 2,
            'tool_calls': [{'function_name': 'bash_command',
                            'arguments': {'keystrokes': 'ls\n', 'duration': 0.1}}],
            'observation': {'results': [{'content': 'app/'}]},
        })
    if overflow_tail:
        trajectory['steps'].append({
            'source': 'agent', 'step_id': 5,
            'tool_calls': [{'function_name': 'bash_command',
                            'arguments': {'keystrokes': 'ls', 'duration': 0.1}}],
            'observation': {'results': [{'content': 'x' * 2048}]},
        })
    (tmp_path / 'trajectory.json').write_text(json.dumps(trajectory))
    (tmp_path / 'result.json').write_text('{}')
    manifest = tmp_path / 'manifest.json'
    manifest.write_text(json.dumps({'records': [{
        'trial': 'trial', 'task': 'task', 'reward': 0,
        'trajectory_file': 'trajectory.json', 'result_file': 'result.json',
    }]}))
    encoding = EncodingConfig()
    config = TrainConfig(outer_lr=5e-4, outer_lr_schedule=schedule)
    def run(name, epochs, resume=None):
        torch.manual_seed(31)
        model, fast = create_model(ModelConfig(hidden_size=8, intermediate_size=12,
            num_layers=1, num_heads=2, num_kv_heads=1, qk_norm=True,
            max_seq_len=1024, dtype='bfloat16'))
        path = tmp_path / name
        train(manifest=manifest, output=path, model=model, initial_fast=fast,
              tokenizer=Tokenizer(), encoding_config=encoding, config=config, epochs=epochs, resume=resume,
              save_every_windows={"window": 1, "window_end": 2}.get(stop_at, 50))
        return path
    module = importlib.import_module('terminal_learner.train')
    save = module.save_checkpoint
    class StopAfterEpoch(Exception):
        pass
    def save_then_stop(*args, **kwargs):
        save(*args, **kwargs)
        raise StopAfterEpoch()
    # Interrupt inside the first window, so the handler must allow it to finish.
    if stop_at.startswith('SIG'):
        from terminal_learner.model.loop import Trainer
        original = Trainer.train_episode
        def interrupting(self, *args, **kwargs):
            step = self.optimizer.step
            def step_and_signal(*a, **kw):
                result = step(*a, **kw)
                os.kill(os.getpid(), getattr(signal, stop_at))
                return result
            self.optimizer.step = step_and_signal
            return original(self, *args, **kwargs)
        previous = signal.getsignal(getattr(signal, stop_at))
        with monkeypatch.context() as patch:
            patch.setattr(Trainer, 'train_episode', interrupting)
            run('first.pt', 2)
        assert signal.getsignal(getattr(signal, stop_at)) == previous
    elif stop_at == 'logging_failure':
        with monkeypatch.context() as patch:
            patch.setattr(module.Tracker, 'log', lambda *a: (_ for _ in ()).throw(RuntimeError('logging failed')))
            with pytest.raises(RuntimeError, match='logging failed'):
                run('first.pt', 2)
    else:
        with monkeypatch.context() as patch:
            patch.setattr(module, 'save_checkpoint', save_then_stop)
            with pytest.raises(StopAfterEpoch):
                run('first.pt', 2)
    if stop_at not in ('epoch', 'window_end'):
        partial = torch.load(tmp_path / 'first.pt', weights_only=True)['training']
        assert partial['windows_completed'] == 1
        assert partial['completed_epochs'] == 0
        assert partial['progress']['episode_state']['next_turn'] == 2
        assert partial['progress']['episode_state']['fast']
    first = tmp_path / 'first.pt'
    # Old checkpoints may still carry this removed setting; it must not reactivate clipping.
    legacy = torch.load(first, weights_only=True)
    legacy['training']['train_config']['outer_clip'] = 1.0
    torch.save(legacy, first)
    resumed = torch.load(run('resumed.pt', 2, first), weights_only=True)
    continuous = torch.load(run('continuous.pt', 2), weights_only=True)
    def same(left, right):
        if isinstance(left, torch.Tensor):
            assert torch.equal(left, right)
        elif isinstance(left, dict):
            assert left.keys() == right.keys()
            for key in left:
                same(left[key], right[key])
        elif isinstance(left, (tuple, list)):
            assert len(left) == len(right)
            for x, y in zip(left, right):
                same(x, y)
        else:
            assert left == right
    same(resumed, continuous)
    assert resumed['training']['optimizer']['fp32_master_weights']
    assert resumed['training']['windows_completed'] == 4
    assert resumed['training']['context_filter']['turn_limits'] == {'trial': 3}
    assert len(resumed['training']['context_filter']['dropped']) == int(overflow_tail)

    if schedule == 'warmup_cosine':
        assert resumed['training']['outer_lr_plan'] == {'total_windows': 4, 'warmup_windows': 1}
        assert resumed['training']['optimizer']['param_groups'][0]['lr'] == pytest.approx(1e-5)

    rows = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    windows = [row for row in rows if row['event'] == 'window']
    assert windows
    for row in windows:
        assert row['window_seconds'] > 0
        assert row['target_tokens_per_second'] > 0
        assert 0 <= row['inner_clip_ratio'] <= 1
        assert row['inner_update_norm'] >= 0
        assert row['inner_planned_update_norm'] >= 0
        assert row['peak_allocated_bytes'] is None
