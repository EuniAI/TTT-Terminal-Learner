import math

import pytest
import torch

from terminal_learner.config import ModelConfig, TrainConfig, EncodingConfig
from terminal_learner.dataloader.encoding import Encoder
from terminal_learner.dataloader.tokenizer import Tokenizer
from terminal_learner.model.data import Episode, Turn
from terminal_learner.model.loop import Trainer
from terminal_learner.model.transformer import create_model
from terminal_learner.protocol import canonical_action


@pytest.mark.parametrize('unroll_steps', [2, 3, 4])
def test_window_size_and_incomplete_tail(unroll_steps):
    torch.set_num_threads(1)
    torch.manual_seed(42)
    model, fast = create_model(ModelConfig(hidden_size=8, intermediate_size=12,
        num_layers=1, num_heads=2, num_kv_heads=2, max_seq_len=1024))
    encoder = Encoder(Tokenizer(), EncodingConfig(), 1024)
    action = canonical_action({'commands': [], 'task_complete': False})
    episode = Episode('task', '$', [Turn(action, '$' * (i + 1), i) for i in range(5)], 'trial', 'task', 0)
    trainer = Trainer(model, fast, encoder, TrainConfig(unroll_steps=unroll_steps))
    windows = []
    result = trainer.train_episode(episode, on_window=windows.append)
    assert result['windows'] == math.ceil(5 / unroll_steps)
    assert result['used_turns'] == 5
    assert [len(row['steps']) for row in windows] == [
        min(unroll_steps, 5 - start) for start in range(0, 5, unroll_steps)]

    all_steps = [step for row in windows for step in row['steps']]
    for row in windows:
        tokens = sum(step['observation_tokens'] for step in row['steps'])
        expected = sum(step['observation_nll'] * step['observation_tokens'] for step in row['steps']) / tokens
        assert row['observation_nll'] == pytest.approx(expected)
    total = sum(step['observation_tokens'] for step in all_steps)
    expected = sum(step['observation_nll'] * step['observation_tokens'] for step in all_steps) / total
    assert result['observation_tokens'] == total
    assert result['observation_nll'] == pytest.approx(expected)


def test_outer_gradients_reach_adamw_without_clipping(monkeypatch):
    import terminal_learner.model.loop as loop
    model, fast = create_model(ModelConfig(hidden_size=8, intermediate_size=12,
        num_layers=1, num_heads=2, num_kv_heads=2, max_seq_len=1024))
    encoder = Encoder(Tokenizer(), EncodingConfig(), 1024)
    trainer = Trainer(model, fast, encoder, TrainConfig())
    def loss_window(model, fast, encoder, episode, turns, history, config):
        # Every parameter receives gradient 10: the global norm exceeds 1.
        loss = sum(p.sum() * 10 for p in model.parameters())
        stats = [{'action_tokens': 1, 'observation_tokens': 0,
                  'observation_truncated': False} for _ in turns]
        return loss, fast, history, stats
    monkeypatch.setattr(loop, 'window_loss', loss_window)
    step = trainer.optimizer.step
    def checked_step():
        for p in trainer.optimizer.master_parameters:
            assert torch.equal(p.grad, torch.full_like(p, 10))
        return step()
    monkeypatch.setattr(trainer.optimizer, 'step', checked_step)
    rows = []
    trainer.train_episode(Episode('task', '$', [Turn('x', None, i) for i in range(2)],
                                  'trial', 'task', 0), on_window=rows.append)
    expected = 10 * math.sqrt(sum(p.numel() for p in model.parameters()))
    assert rows[0]['outer_grad_norm'] == pytest.approx(expected)

    assert rows[0]['observation_nll'] is None
    assert rows[0]['observation_tokens'] == 0
