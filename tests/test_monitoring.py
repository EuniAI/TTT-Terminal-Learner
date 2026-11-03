import sys
from types import SimpleNamespace

import pytest
import torch

from terminal_learner.monitoring import Tracker
from terminal_learner.model.meta_model import inner_update
from terminal_learner.cli import parse_args


def test_inner_metrics_do_not_change_differentiable_update():
    fast = {'w': torch.tensor([3., 4.], requires_grad=True)}
    updated, metrics = inner_update(fast, fast['w'].square().sum() / 2,
                                  lr=.1, create_graph=True, return_stats=True)
    assert metrics['inner_grad_norm'] == pytest.approx(5)
    assert metrics['inner_update_norm'] == pytest.approx(.1)
    assert metrics['inner_planned_update_norm'] == pytest.approx(.1)
    assert metrics['inner_clipped']
    assert torch.autograd.grad(updated['w'].sum(), fast['w'])[0].isfinite().all()


def test_optional_tracker_only_logs_scalars(monkeypatch):
    calls = []
    fake = SimpleNamespace(id='test-id', project='test', entity='team',
                           define_metric=lambda *a, **kw: None, log=lambda row, **kw: calls.append((row, kw)),
                           finish=lambda **kw: calls.append('finish'))
    monkeypatch.setitem(sys.modules, 'wandb', SimpleNamespace(init=lambda **kw: fake, Settings=lambda **kw: kw))
    tracker = Tracker(mode='offline', project='test')
    tracker.log({'window': 3, 'action_nll': 1.2, 'trial': 'private',
                 'steps': [{'output': 'private'}], 'peak_allocated_bytes': None})
    tracker.close()
    assert calls == [({'window': 3, 'action_nll': 1.2}, {}), 'finish']
    monkeypatch.setitem(sys.modules, 'wandb', None)
    Tracker().close()


def test_monitoring_cli():
    config = parse_args(['--tiny', '--save-every-windows', '7',
                         '--wandb-mode', 'offline', '--wandb-project', 'test']).config
    assert config.run.save_every_windows == 7
    assert config.run.wandb_project == 'test'
    with pytest.raises(SystemExit):
        parse_args(['--tiny', '--save-every-windows', '0'])


def test_bf16_planned_and_actual_update_differ_without_changing_gradients():
    fast = {'w': torch.tensor([1.], dtype=torch.bfloat16, requires_grad=True)}
    def update(stats):
        return inner_update(fast, fast['w'].float().square().sum() / 2,
                            lr=.001, create_graph=True, clip_norm=2, return_stats=stats)
    plain = update(False)
    tracked, metrics = update(True)
    assert metrics['inner_planned_update_norm'] == pytest.approx(.001)
    assert metrics['inner_update_norm'] == 0
    assert torch.equal(plain['w'], tracked['w'])
    assert torch.equal(torch.autograd.grad(plain['w'].sum(), fast['w'])[0],
                       torch.autograd.grad(tracked['w'].sum(), fast['w'])[0])
