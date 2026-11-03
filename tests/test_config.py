from pathlib import Path

import pytest
import yaml

from terminal_learner.cli import CONFIG_DIRECTORY, parse_args
from terminal_learner.config import RunConfig


def test_full_training_config():
    config = RunConfig.load(CONFIG_DIRECTORY / 'terminal_ttt.yaml')
    assert config.model.model_id == "Qwen/Qwen3-8B"
    assert config.model.revision == "b968826d9c46dd6066d109eabc6255188de91218"
    assert config.model.attention_backend == 'flashbob'
    assert parse_args(['--tiny']).config.model.attention_backend == 'math'
    assert parse_args(['--attention-backend', 'math']).config.model.attention_backend == 'math'
    assert config.model.max_seq_len is None
    assert parse_args(["--max-seq-len", "8192"]).config.model.max_seq_len == 8192
    assert parse_args(["--max-seq-len", "auto"]).config.model.max_seq_len is None
    assert config.model.adaptive_layers == 9
    assert config.training.outer_lr_schedule == 'warmup_cosine'
    assert config.training.outer_lr == 5e-4
    assert config.training.outer_end_lr == 1e-5
    assert config.training.unroll_steps == 2
    assert config.training.action_supervision == 'all'
    assert config.encoding.enable_thinking is True
    assert parse_args(['--no-enable-thinking']).config.encoding.enable_thinking is False
    assert config.data.test_fraction == 0
    assert config.run.limit_episodes is config.run.limit_turns is None
    assert Path(config.data.manifest).name == 'expert-manifest.json'
    assert (CONFIG_DIRECTORY.parents[1] / 'scripts/prepare_data.py').is_file()


def test_yaml_paths_are_relative_to_file(tmp_path, monkeypatch):
    path = tmp_path / 'experiment.yaml'
    path.write_text('data:\n  manifest: data.json\nrun:\n  output: model.pt\n')
    monkeypatch.chdir('/')
    config = RunConfig.load(path)
    assert config.data.manifest == str(tmp_path / 'data.json')
    assert config.run.output == str(tmp_path / 'model.pt')


def test_yaml_and_cli_precedence(tmp_path):
    path = tmp_path / 'experiment.yaml'
    path.write_text('training:\n  unroll_steps: 4\n')
    assert parse_args(['--config', str(path)]).config.training.unroll_steps == 4
    args = parse_args(['--config', str(path), '--unroll-steps', '3'])
    assert args.config.training.unroll_steps == 3
    assert parse_args(['--tiny']).config.model.model_id == 'tiny-random'
    args = parse_args(['--model', 'another/model'])
    assert args.config.model.revision is None


@pytest.mark.parametrize('data', [
    {'training': {'unroll_steps': 1}},
    {'training': {'unroll_steps': True}},
    {'training': {'unroll_steps': 2.5}},
    {'training': {'unroll_steps': '4'}},
    {'training': {'unroll_step': 4}},
    {'train': {}},
    {'model': {'attention_backend': 'unknown'}},
    {'encoding': {'loss_chunk_tokens': 4096}},
    {'run': {'limit_turns': 1}},
    {'training': {'outer_lr': float('nan')}},
])
def test_invalid_yaml_rejected(tmp_path, data):
    path = tmp_path / 'experiment.yaml'
    path.write_text(yaml.safe_dump(data))
    with pytest.raises(ValueError):
        RunConfig.load(path)



def test_wandb_management_yaml_and_cli(tmp_path):
    path = tmp_path / 'experiment.yaml'
    path.write_text('run:\n  wandb_mode: offline\n  wandb_project: test\n  wandb_entity: team\n  wandb_dir: logs\n  wandb_new_run: true\n')
    config = RunConfig.load(path)
    assert config.run.wandb_dir == str(tmp_path / 'logs')
    assert config.run.wandb_new_run is True
    parsed = parse_args(['--config', str(path), '--no-wandb-new-run', '--wandb-dir', 'relative', '--wandb-entity', 'other']).config
    assert parsed.run.wandb_new_run is False
    assert parsed.run.wandb_dir == 'relative' and parsed.run.wandb_entity == 'other'
    assert parse_args(['--config', str(path), '--wandb-new-run']).config.run.wandb_new_run is True
    with pytest.raises(SystemExit):
        parse_args(['--tiny', '--wandb-new-run'])
