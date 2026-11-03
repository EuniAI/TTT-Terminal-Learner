import os

import pytest
import torch

from terminal_learner import checkpoint
from terminal_learner.cli import parse_args


def write(path, step, keep=3):
    checkpoint.retained_atomic_save({'step': step}, path, window=step, keep=keep)


def steps(path):
    return sorted(torch.load(p, weights_only=True)['step']
                  for p in path.parent.joinpath(path.stem + '.checkpoints').glob('window-*.pt'))


def test_retains_three_and_latest_is_hardlink(tmp_path):
    path = tmp_path / 'agent.pt'
    for step in (50, 100, 150, 200):
        write(path, step)
    assert steps(path) == [100, 150, 200]
    latest = tmp_path / 'agent.checkpoints/window-000000000200.pt'
    assert os.path.samefile(path, latest)
    assert torch.load(path, weights_only=True)['step'] == 200
    write(path, 200)
    assert steps(path) == [100, 150, 200]
    write(path, 250, keep=1)
    assert steps(path) == [250]
    assert parse_args(['--tiny', '--keep-checkpoints', '2']).config.run.keep_checkpoints == 2


def test_resuming_lower_window_keeps_new_save(tmp_path):
    path = tmp_path / 'agent.pt'
    for step in (100, 200, 300):
        write(path, step)
        snapshot = tmp_path / f'agent.checkpoints/window-{step:012d}.pt'
        os.utime(snapshot, ns=(step, step))
    write(path, 50)
    assert steps(path) == [50, 200, 300]
    assert torch.load(path, weights_only=True)['step'] == 50


@pytest.mark.parametrize('failure', ['write', 'publish'])
def test_failure_does_not_replace_latest_or_prune_old(tmp_path, monkeypatch, failure):
    path = tmp_path / 'agent.pt'
    for step in (1, 2, 3):
        write(path, step)
    def fail(*args, **kwargs):
        raise OSError('simulated disk failure')
    if failure == 'write':
        monkeypatch.setattr(checkpoint.torch, 'save', fail)
    else:
        monkeypatch.setattr(checkpoint.os, 'link', fail)
    with pytest.raises(OSError, match='simulated'):
        write(path, 4)
    assert torch.load(path, weights_only=True)['step'] == 3
    assert {1, 2, 3}.issubset(steps(path))
    assert not list(tmp_path.rglob('*.tmp'))
    assert not list(tmp_path.glob('*.link'))


def test_failure_resaving_same_window_preserves_old_latest(tmp_path, monkeypatch):
    path = tmp_path / 'agent.pt'
    write(path, 3)
    original = path.read_bytes()
    def fail(*args):
        raise OSError('publication failed')
    monkeypatch.setattr(checkpoint.os, 'link', fail)
    with pytest.raises(OSError):
        checkpoint.retained_atomic_save({'step': 3, 'new': True}, path, window=3)
    assert path.read_bytes() == original
