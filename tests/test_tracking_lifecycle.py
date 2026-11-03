import importlib
import json
import sys
from types import SimpleNamespace

import pytest
import torch

from terminal_learner.config import EncodingConfig, ModelConfig
from terminal_learner.dataloader.tokenizer import Tokenizer
from terminal_learner.model.transformer import create_model
from terminal_learner.monitoring import Tracker
from terminal_learner.train import train


@pytest.fixture
def sdk(monkeypatch):
    calls, runs = [], []
    def init(**kwargs):
        calls.append(kwargs)
        def log(row):
            run.logs.append(row)
            if run.fail and 'action_nll' in row:
                raise RuntimeError('upload failed')
        def finish(**kwargs):
            if run.fail:
                raise RuntimeError('finish failed')
        run = SimpleNamespace(id=kwargs['id'], project=kwargs['project'], entity=kwargs['entity'] or 'team',
                              logs=[], fail=False, define_metric=lambda *a, **kw: None,
                              log=log, finish=finish)
        runs.append(run)
        return run
    monkeypatch.setitem(sys.modules, 'wandb', SimpleNamespace(init=init, Settings=lambda **kw: kw))
    return calls, runs


def test_tracker_identity_modes(sdk):
    calls, _ = sdk
    first = Tracker(mode='online', project='project')
    assert calls[-1]['resume'] == 'never'
    identity = first.identity
    first.close()
    disabled = Tracker(previous=identity)
    assert disabled.identity == identity and len(calls) == 1
    resumed = Tracker(mode='online', project='project', previous=disabled.identity)
    assert calls[-1]['id'] == identity['id']
    assert calls[-1]['entity'] == 'team'
    assert calls[-1]['resume'] == 'must'
    resumed.close()
    with pytest.raises(ValueError, match='project differs'):
        Tracker(mode='online', project='other', previous=identity)
    offline = Tracker(mode='offline', project='project', previous=identity)
    assert calls[-1]['id'] != identity['id']
    assert calls[-1]['resume'] is None
    assert calls[-1]['config']['parent_run_id'] == identity['id']
    offline.close()


def test_existing_output_rejected_before_dataset_access(tmp_path):
    path = tmp_path / 'model.pt'
    path.write_bytes(b'old training')
    with pytest.raises(FileExistsError, match='--resume'):
        train(manifest=tmp_path / 'missing.json', output=path)
    assert path.read_bytes() == b'old training'


@pytest.mark.parametrize('interval', [1, 50])
def test_logging_failure_checkpoint_and_online_resume(tmp_path, monkeypatch, capsys, sdk, interval):
    calls, runs = sdk
    torch.set_num_threads(1)
    trajectory = {'agent': {'name': 'terminus-2', 'model_name': 'anthropic/claude-opus-4-6'},
        'steps': [{'source': 'user', 'message': 'Task Description:\nlist files\nCurrent terminal state:$'}]}
    for i in range(3):
        trajectory['steps'].append({'source': 'agent', 'step_id': i+2,
            'tool_calls': [{'function_name': 'bash_command', 'arguments': {'keystrokes': 'ls\n', 'duration': .1}}],
            'observation': {'results': [{'content': 'app/'}]}})
    (tmp_path / 'trajectory.json').write_text(json.dumps(trajectory))
    (tmp_path / 'result.json').write_text('{}')
    manifest = tmp_path / 'manifest.json'
    manifest.write_text(json.dumps({'records': [{'trial': 'trial', 'task': 'task', 'reward': 0,
        'trajectory_file': 'trajectory.json', 'result_file': 'result.json'}]}))
    model, fast = create_model(ModelConfig(hidden_size=8, intermediate_size=12, num_layers=1,
        num_heads=2, num_kv_heads=1, max_seq_len=1024))
    path = tmp_path / 'agent.pt'
    module = importlib.import_module('terminal_learner.train')
    original = module.save_checkpoint
    saved = []
    def save(*args, **kwargs):
        original(*args, **kwargs)
        saved.append(kwargs['training']['windows_completed'])
        if len(saved) == 1:
            assert kwargs['training']['wandb']['id'] == calls[0]['id']
            runs[-1].fail = True
    with monkeypatch.context() as patch:
        patch.setattr(module, 'save_checkpoint', save)
        with pytest.warns(RuntimeWarning, match='finish failed'):
            with pytest.raises(RuntimeError, match='upload failed'):
                train(manifest=manifest, output=path, model=model, initial_fast=fast,
                    tokenizer=Tokenizer(), encoding_config=EncodingConfig(),
                    save_every_windows=interval, wandb_mode='online', wandb_project='project')
    assert saved == [0, 1]  # No duplicate save if the periodic checkpoint already succeeded.
    partial = torch.load(path, weights_only=True)['training']
    assert partial['progress']['episode_state']['next_turn'] == 2
    train(manifest=manifest, output=path, resume=path, wandb_mode='online', wandb_project='project')
    complete = torch.load(path, weights_only=True)['training']
    assert calls[1]['id'] == calls[0]['id']
    assert calls[1]['resume'] == 'must'
    assert complete['wandb'] == partial['wandb']
    assert complete['windows_completed'] == 2 and complete['completed_epochs'] == 1

    rows = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    for row in rows:
        if row['event'] == 'checkpoint':
            assert row['checkpoint_seconds'] > 0 and row['checkpoint_saved'] == 1
        if row['event'] == 'window':
            assert row['checkpoint_seconds'] >= 0
            assert row['observation_nll'] > 0
    assert any(r.get('checkpoint_seconds', 0) > 0 for r in runs[-1].logs)
    assert any('observation_nll' in r for r in runs[-1].logs)


def test_tracker_fork_entity_and_directory(sdk, tmp_path):
    calls, _ = sdk
    original = Tracker(mode='online', project='original', entity='old-team')
    identity = original.identity
    original.close()
    with pytest.raises(ValueError, match='entity differs'):
        Tracker(mode='online', project='original', entity='new-team', previous=identity)
    directory = tmp_path / 'logs'
    fork = Tracker(mode='online', project='new-project', entity='new-team',
                   directory=directory, previous=identity, new_run=True)
    call = calls[-1]
    assert call['id'] != identity['id'] and call['resume'] == 'never'
    assert call['config']['parent_run_id'] == identity['id']
    assert call['dir'] == str(directory.resolve()) and directory.is_dir()
    assert fork.identity['entity'] == 'new-team'
    fork.close()
    continued = Tracker(mode='online', previous=fork.identity)
    assert calls[-1]['id'] == fork.identity['id'] and calls[-1]['resume'] == 'must'
    continued.close()
