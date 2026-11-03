from dataclasses import replace
from types import SimpleNamespace
import importlib

import pytest
import torch

from terminal_learner.config import TrainConfig, ModelConfig
from terminal_learner.schedule import make_schedule, outer_learning_rate
from terminal_learner.model.transformer import TerminalTransformer


def test_warmup_peak_cosine_end_and_extension():
    config = TrainConfig(outer_lr=5e-4, outer_lr_schedule='warmup_cosine',
                         outer_warmup_ratio=0.1, outer_end_lr=1e-5)
    plan = make_schedule(config, 50, 2)
    assert plan == {'total_windows': 100, 'warmup_windows': 10}
    rates = [outer_learning_rate(config, i, plan) for i in range(110)]
    assert rates[0] == 0
    assert rates[5] == pytest.approx(2.5e-4)
    assert rates[10] == pytest.approx(5e-4)
    assert all(a >= b for a, b in zip(rates[10:], rates[11:]))
    assert rates[99] == rates[109] == pytest.approx(1e-5)
    assert outer_learning_rate(TrainConfig(), 100, None) == 1e-5
    with pytest.raises(ValueError, match='at least 3'):
        make_schedule(config, 1, 2)
    assert make_schedule(config, 3, 1)['warmup_windows'] == 1
    with pytest.raises(ValueError, match='saved outer_lr_plan'):
        outer_learning_rate(config, 2, None)
    for bad in [replace(config, outer_end_lr=1), replace(config, outer_warmup_ratio=1),
                replace(config, outer_warmup_ratio=float('nan'))]:
        with pytest.raises(ValueError):
            bad.validate()


def test_window_count_includes_each_episode_tail_and_selection(monkeypatch):
    module = importlib.import_module('terminal_learner.train')
    monkeypatch.setattr(module, 'load_episode', lambda r: SimpleNamespace(turns=[None] * r['n']))
    records = [{'n': 3, 'reward': 0}, {'n': 3, 'reward': 1}, {'n': 0, 'reward': 1}]
    assert module.count_training_windows(records, TrainConfig()) == 4  # Not ceil(6/2).
    assert module.count_training_windows(records, TrainConfig(), limit_turns=2) == 2
    assert module.count_training_windows(records, TrainConfig(action_supervision='success')) == 2


def test_last_nine_of_36_layers_without_allocating_8b():
    with torch.device('meta'):
        model = TerminalTransformer(ModelConfig(num_layers=36, adaptive_layers=9))
    assert model.split == 27
    assert [i for i, layer in enumerate(model.layers) if layer.prime_norm is not None] == list(range(27, 36))
    assert len(model.fast_shapes()) == 27
