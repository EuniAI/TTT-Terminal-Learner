import json

import pytest

from terminal_learner.config import EncodingConfig, TrainConfig
from terminal_learner.dataloader.context import context_turn_limits, apply_context_limits
from terminal_learner.dataloader.encoding import Encoder
from terminal_learner.dataloader.tokenizer import Tokenizer
from terminal_learner.dataloader.trajectory import load_episode
from terminal_learner.distributed_train import total_updates
from terminal_learner.train import count_training_windows


def record(tmp_path, outputs):
    path = tmp_path / 'trajectory.json'
    path.write_text(json.dumps({
        'agent': {'name': 'terminus-2', 'model_name': 'anthropic/claude-opus-4-6'},
        'steps': [{'source': 'user', 'message': 'task'}] + [
            {'source': 'agent', 'step_id': i + 2,
             'tool_calls': [{'function_name': 'bash_command',
                             'arguments': {'keystrokes': 'ls', 'duration': 1}}],
             'observation': {'results': [{'content': output}]}}
            for i, output in enumerate(outputs)]}))
    return {'trajectory_file': str(path), 'trial': 'trial', 'task': 'task', 'reward': 0}


def test_observation_overflow_drops_whole_turn_and_suffix(tmp_path):
    r = record(tmp_path, ['ok', 'x' * 2000, 'short again'])
    original = open(r['trajectory_file']).read()
    enc = Encoder(Tokenizer(), EncodingConfig(), 1024)
    limits, dropped = context_turn_limits([r], enc)
    assert limits == {'trial': 1}
    assert dropped[0]['kind'] == 'observation'
    assert dropped[0]['dropped_turns'] == 2
    selected = apply_context_limits([r], limits)
    assert len(load_episode(selected[0]).turns) == 1
    assert len(load_episode(r).turns) == 3
    assert open(r['trajectory_file']).read() == original
    assert count_training_windows(selected, TrainConfig()) == 1
    assert total_updates(selected, TrainConfig(), None, 2, 0, 2) == 2


def test_exact_boundary_empty_prefix_and_diagnostic_limit(tmp_path):
    r = record(tmp_path, ['ok', 'x' * 2000])
    enc = Encoder(Tokenizer(), EncodingConfig(), 1024)
    e = load_episode(r)
    h = enc.initial_history(e.initial_observation)
    p, t = enc.observation_example(e.task, h, e.turns[0].action, 'ok')
    boundary = len(p) + len(t)
    enc.max_seq_len = boundary
    limits, dropped = context_turn_limits([r], enc, limit_turns=1)
    assert limits == {'trial': 1} and not dropped
    enc.max_seq_len = boundary - 1
    limits, dropped = context_turn_limits([r], enc)
    assert limits == {'trial': 0} and dropped[0]['dropped_turns'] == 2
    assert count_training_windows(apply_context_limits([r], limits), TrainConfig()) == 0
    enc.max_seq_len = 2
    _, dropped = context_turn_limits([r], enc)
    assert dropped[0]['kind'] == 'action'
    with pytest.raises(ValueError, match='do not match'):
        apply_context_limits([r], {})
