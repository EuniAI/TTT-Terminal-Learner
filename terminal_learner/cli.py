"""YAML experiment selection and explicit command-line overrides."""
from __future__ import annotations

import argparse
import yaml
from dataclasses import fields
from pathlib import Path

from .config import RunConfig
from .train import train

CONFIG_DIRECTORY = Path(__file__).resolve().parents[1] / 'configs' / 'experiment'


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, argument_default=argparse.SUPPRESS)
    parser.add_argument('--config', help='experiment YAML; defaults to terminal_ttt.yaml')
    parser.add_argument('--self-test', action='store_true', help='run the pytest suite')
    source = parser.add_mutually_exclusive_group()
    source.add_argument('--model', dest='model_id')
    source.add_argument('--tiny', action='store_true', help='select tiny.yaml unless --config is provided')
    source.add_argument('--resume', help='restore model, optimizer and training settings from checkpoint')
    defaults = RunConfig()
    for section in ('model', 'training', 'encoding', 'data', 'run'):
        for item in fields(getattr(defaults, section)):
            if item.name == 'model_id':
                continue
            value = getattr(getattr(defaults, section), item.name)
            kind = int if item.name in ('limit_episodes', 'limit_turns') else type(value)
            if value is None:
                kind = int if item.name.startswith('limit_') else str
            if item.name == 'max_seq_len':
                kind = lambda value: None if value.lower() == 'auto' else int(value)
            if item.name == 'model_parallel_devices':
                kind = lambda value: None if value.lower() == 'none' else value.split(',')
            if item.name == 'model_parallel_layer_counts':
                kind = lambda value: None if value.lower() == 'auto' else [int(v) for v in value.split(',')]
            if kind is bool:
                parser.add_argument('--' + item.name.replace('_', '-'), action=argparse.BooleanOptionalAction)
            else:
                parser.add_argument('--' + item.name.replace('_', '-'), type=kind)
    options = vars(parser.parse_args(argv))
    path = options.pop('config', str(CONFIG_DIRECTORY / ('tiny.yaml' if options.get('tiny') else 'terminal_ttt.yaml')))
    tiny = options.pop('tiny', False)
    resume = options.pop('resume', None)
    self_test = options.pop('self_test', False)
    try:
        config = RunConfig.load(path)
        if tiny:
            config.model.model_id, config.model.revision = 'tiny-random', None
        if 'model_id' in options and options['model_id'] != config.model.model_id and 'revision' not in options:
            config.model.revision = None
        for section in ('model', 'training', 'encoding', 'data', 'run'):
            obj = getattr(config, section)
            for item in fields(obj):
                if item.name in options:
                    setattr(obj, item.name, options[item.name])
        config.validate()
    except (OSError, ValueError, TypeError, yaml.YAMLError) as error:
        parser.error(f'{path}: {error}')
    return argparse.Namespace(config=config, resume=resume, self_test=self_test)


def main():
    args = parse_args()
    if args.self_test:
        import pytest
        raise SystemExit(pytest.main([str(Path(__file__).resolve().parents[1] / 'tests'), '-q']))
    c = args.config
    train(manifest=c.data.manifest, output=c.run.output,
          model_id=c.model.model_id, revision=c.model.revision,
          tiny=c.model.model_id == 'tiny-random', device=c.run.device,
          model_parallel_devices=c.run.model_parallel_devices,
          model_parallel_layer_counts=c.run.model_parallel_layer_counts,
          distributed_timeout_seconds=c.run.distributed_timeout_seconds,
          distributed_bucket_mb=c.run.distributed_bucket_mb,
          dtype=c.model.dtype, adaptive_layers=c.model.adaptive_layers,
          attention_backend=c.model.attention_backend,
          max_seq_len=c.model.max_seq_len, encoding_config=c.encoding,
          config=c.training, epochs=c.run.epochs, seed=c.run.seed,
          test_fraction=c.data.test_fraction, resume=args.resume,
          limit_episodes=c.run.limit_episodes, limit_turns=c.run.limit_turns,
          keep_checkpoints=c.run.keep_checkpoints, save_every_windows=c.run.save_every_windows,
          wandb_mode=c.run.wandb_mode, wandb_project=c.run.wandb_project, wandb_name=c.run.wandb_name,
          wandb_entity=c.run.wandb_entity, wandb_dir=c.run.wandb_dir, wandb_new_run=c.run.wandb_new_run)
