from __future__ import annotations

"""Dataset orchestration and window-boundary checkpoint/resume."""
import json
import math
import random
import sys
import time
from dataclasses import asdict
from pathlib import Path
import torch
from .monitoring import Tracker, stop_requests
from .config import ModelConfig, EncodingConfig, TrainConfig
from .checkpoint import load_checkpoint, save_checkpoint
from .dataloader.encoding import Encoder
from .dataloader.tokenizer import Tokenizer
from .dataloader.trajectory import read_manifest, load_episode, split_tasks
from .model.transformer import create_model
from .model.pretrained import import_pretrained
from .model.loop import Trainer
from .schedule import make_schedule, outer_learning_rate

def _train(*, manifest, output, model=None, initial_fast=None, tokenizer=None,
          model_id=None, revision=None, tiny=False, device='cpu', dtype='float32',
          adaptive_layers=1, max_seq_len=None, encoding_config=None, config=None,
          epochs=1, seed=0, test_fraction=0.0, resume=None,
          limit_episodes=None, limit_turns=None, attention_backend=None, keep_checkpoints=3,
          save_every_windows=50, wandb_mode="disabled", wandb_project=None, wandb_name=None,
          wandb_entity=None, wandb_dir=None, wandb_new_run=False,
          distributed_timeout_seconds=1800, distributed_bucket_mb=25,
          model_parallel_devices=None, model_parallel_layer_counts=None,
          _prepare_only=False, _parallel=None):
    """Train with periodic, epoch-end, and cooperative-stop checkpoints.

    resume restores optimizer/RNG/epoch counters; target epochs is cumulative.
    Resume uses stored hyperparameters, encoding, split, and diagnostic limits.
    """
    if model_parallel_devices is not None:
        from .model.placement import validate_devices
        validate_devices(model_parallel_devices)
        if _parallel is not None:
            raise ValueError('model parallel training uses one process, not torchrun')
        device = model_parallel_devices[0]
    if type(save_every_windows) is not int or save_every_windows < 1:
        raise ValueError('save_every_windows must be a positive integer')
    if (type(distributed_timeout_seconds) is not int or distributed_timeout_seconds < 1
            or type(distributed_bucket_mb) is not int or distributed_bucket_mb < 1):
        raise ValueError('distributed timeout and bucket size must be positive integers')
    if type(keep_checkpoints) is not int or keep_checkpoints < 1:
        raise ValueError('keep_checkpoints must be a positive integer')
    if epochs < 1 or (limit_episodes is not None and limit_episodes < 1):
        raise ValueError('epochs and limit_episodes must be positive')
    output_path = Path(output)
    if not resume and (output_path.exists() or output_path.is_symlink()):
        raise FileExistsError('output checkpoint already exists; use --resume or a new output')
    records, fingerprint = read_manifest(manifest)
    torch.manual_seed(seed)
    config = config or TrainConfig()
    encoding_config = encoding_config or EncodingConfig()
    completed_epochs = 0
    prior = None
    if resume:
        model, initial_fast, encoder, payload = load_checkpoint(resume, device, attention_backend=attention_backend,
            devices=model_parallel_devices, layer_counts=model_parallel_layer_counts)
        prior = payload['training']
        if prior.get('model_parallel') != getattr(model, 'layer_placement', None):
            raise ValueError('training resume requires the saved model parallel layout; inference can use another layout')
        if prior.get('cuda_rng') and len(prior['cuda_rng']) != torch.cuda.device_count():
            raise ValueError('training resume requires the same number of visible CUDA devices')
        if prior.get('distributed') and _parallel is None:
            raise ValueError('distributed training checkpoint must resume with the same torchrun world size; inference is supported on one device')
        if _parallel is not None:
            if not prior.get('distributed') or prior['distributed']['world_size'] != _parallel.world_size:
                raise ValueError('distributed resume requires a checkpoint with the same world size')
        if prior.get('data_fingerprint') != fingerprint:
            raise ValueError('resume dataset differs from checkpoint')
        stored_config = dict(prior['train_config'])
        stored_config.pop('outer_clip', None)  # Legacy checkpoints: outer clipping is removed.
        config = TrainConfig(**stored_config)
        tokenizer, encoding_config = encoder.tok, encoder.config
        train_tasks, test_tasks = prior['train_tasks'], prior['test_tasks']
        completed_epochs, seed = prior['completed_epochs'], prior['seed']
        limit_episodes, limit_turns = prior['limit_episodes'], prior['limit_turns']
        if epochs <= completed_epochs:
            raise ValueError('epochs must exceed completed checkpoint epochs')
    else:
        train_tasks, test_tasks = split_tasks(records, test_fraction=test_fraction, seed=seed)
        if model is None:
            if tiny:
                tokenizer = Tokenizer.native_tiny()
                model, initial_fast = create_model(ModelConfig(
                    vocab_size=tokenizer.vocab_size,
                    max_seq_len=ModelConfig().max_seq_len if max_seq_len is None else max_seq_len,
                    dtype=dtype, adaptive_layers=adaptive_layers,
                    attention_backend=attention_backend or 'math'), device,
                    devices=model_parallel_devices, layer_counts=model_parallel_layer_counts)
            else:
                if not model_id:
                    raise ValueError('choose --model or --tiny explicitly')
                model, initial_fast, tokenizer = import_pretrained(
                    model_id, revision=revision, adaptive_layers=adaptive_layers,
                    max_seq_len=max_seq_len, dtype=dtype, device=device,
                    attention_backend=attention_backend or 'math',
                    devices=model_parallel_devices, layer_counts=model_parallel_layer_counts)
        if initial_fast is None or tokenizer is None:
            raise ValueError('custom models require initial_fast and tokenizer')
        encoder = Encoder(tokenizer, encoding_config, model.config.max_seq_len)
    if model_parallel_devices is not None:
        if not hasattr(model, 'layer_placement'):
            raise ValueError('custom models must already have model parallel placement')
        print(json.dumps({'event': 'model_placement', 'model_parallel': model.layer_placement}), flush=True)
    if set(train_tasks) & set(test_tasks):
        raise ValueError('train/test task leakage')
    if attention_backend is not None and model.config.attention_backend != attention_backend:
        model.config.attention_backend = attention_backend
        model.configure_backend()
    selected = sorted((r for r in records if r['task'] in set(train_tasks)), key=lambda r: r['trial'])
    if limit_episodes is not None:
        selected = selected[:limit_episodes]
    if prior:
        lr_plan = prior.get('outer_lr_plan')
    elif config.outer_lr_schedule == 'constant':
        lr_plan = None
    else:
        if _parallel is None:
            windows_per_epoch = count_training_windows(selected, config, limit_turns=limit_turns)
            lr_plan = make_schedule(config, windows_per_epoch, epochs)
        else:
            from .distributed_train import total_updates
            lr_plan = make_schedule(config, total_updates(selected, config, limit_turns, epochs, seed,
                                                          _parallel.world_size), 1)
    outer_learning_rate(config, prior['windows_completed'] if prior else 0, lr_plan)
    trainer = Trainer(model, initial_fast, encoder, config, lr_plan=lr_plan)
    if prior:
        trainer.optimizer.load_state_dict(prior['optimizer'])
        trainer.windows_completed = prior['windows_completed']
        if _parallel is None:
            torch.set_rng_state(prior['torch_rng'].cpu())
            if device.startswith('cuda') and prior.get('cuda_rng'):
                torch.cuda.set_rng_state_all(prior['cuda_rng'])
    if _prepare_only:
        from types import SimpleNamespace
        return SimpleNamespace(**locals())
    print(json.dumps({'event': 'dataset', 'retained': len(records), 'train_tasks': len(train_tasks),
                      'test_tasks': len(test_tasks), 'train_episodes': len(selected),
                      'action_supervision': config.action_supervision,
                      'model': model.config.model_id, 'max_seq_len': model.config.max_seq_len,
                      'protocol': tokenizer.protocol,
                      'enable_thinking': encoding_config.enable_thinking,
                      'attention_backend': model.config.attention_backend,
                      'outer_lr_schedule': config.outer_lr_schedule, 'outer_lr_plan': lr_plan}), flush=True)
    progress = prior.get('progress') if prior else None
    cursor = progress or {'episode_index': 0, 'episode_state': None}
    epoch = completed_epochs

    def checkpoint(reason, *, report=True):
        started = time.perf_counter()
        active = cursor['episode_state']
        if active is not None:
            active = {**active, 'fast': {k: v.detach().cpu() for k, v in active['fast'].items()}}
        training = {'manifest': str(Path(manifest).resolve()), 'data_fingerprint': fingerprint,
                    'train_tasks': train_tasks, 'test_tasks': test_tasks,
                    'train_config': asdict(config), 'completed_epochs': epoch,
                    'seed': seed, 'limit_episodes': limit_episodes, 'limit_turns': limit_turns,
                    'windows_completed': trainer.windows_completed, 'outer_lr_plan': lr_plan,
                    'wandb': tracker.identity,
                    'progress': {'episode_index': cursor['episode_index'], 'episode_state': active},
                    'optimizer': trainer.optimizer.state_dict(), 'torch_rng': torch.get_rng_state(),
                    'model_parallel': getattr(model, 'layer_placement', None),
                    'cuda_rng': torch.cuda.get_rng_state_all() if str(device).startswith('cuda') else []}
        save_checkpoint(output, model, initial_fast, tokenizer, encoding_config,
                        inner_lr=config.inner_lr, inner_clip=config.inner_clip, training=training,
                        keep=keep_checkpoints)
        seconds = time.perf_counter() - started
        row = {'event': 'checkpoint', 'path': str(output), 'completed_epochs': epoch,
               'window': trainer.windows_completed, 'reason': reason,
               'checkpoint_seconds': seconds, 'checkpoint_saved': 1}
        print(json.dumps(row), flush=True)
        if report and tracker.run is not None:
            tracker.log(row)  # Already safely saved if reporting fails.
        return seconds

    # Tracker initialization must not perturb training RNG (also on resume).
    with torch.random.fork_rng(devices=list(range(torch.cuda.device_count())) if str(device).startswith('cuda') else []):
        tracker = Tracker(mode=wandb_mode, project=wandb_project, name=wandb_name,
                          config={'training': asdict(config), 'model': asdict(model.config),
                                  'encoding': asdict(encoding_config)},
                          previous=(prior or {}).get('wandb'), entity=wandb_entity,
                          directory=wandb_dir or Path(output).parent, new_run=wandb_new_run)
    try:
        with stop_requests() as stop:
            if tracker.run is not None:
                checkpoint('tracking_start')  # Persist run identity before the first update.
            while epoch < epochs:
                order = list(selected)
                random.Random(seed + epoch).shuffle(order)
                index = cursor['episode_index']
                if not 0 <= index <= len(order):
                    raise ValueError('invalid checkpoint episode index')
                trained = index > 0 or cursor['episode_state'] is not None
                while index < len(order) and not stop['signal']:
                    episode = load_episode(order[index])
                    turn_count = len(episode.turns[:limit_turns]) if limit_turns is not None else len(episode.turns)

                    def on_window(row):
                        state = trainer.episode_state
                        if state['next_turn'] == turn_count:
                            cursor.update(episode_index=index + 1, episode_state=None)
                        else:
                            cursor.update(episode_index=index, episode_state=state)
                        saved = trainer.windows_completed % save_every_windows == 0
                        seconds = checkpoint('periodic', report=False) if saved else 0.0
                        row.update(checkpoint_seconds=seconds, checkpoint_saved=int(saved))
                        print(json.dumps({'event': 'window', 'epoch': epoch + 1, **row}), flush=True)
                        try:
                            tracker.log(row)
                        except Exception:
                            if not saved:
                                checkpoint('logging_failure', report=False)
                            raise

                    result = trainer.train_episode(episode, limit_turns=limit_turns,
                        state=cursor['episode_state'], on_window=on_window,
                        should_stop=lambda: bool(stop['signal']))
                    trained |= 'skipped' not in result
                    print(json.dumps({'event': 'episode', 'epoch': epoch + 1, **result}), flush=True)
                    if stop['signal']:
                        break
                    index += 1
                    cursor.update(episode_index=index, episode_state=None)
                # A checkpoint at the last window can resume with index == len(order).
                if cursor['episode_index'] == len(order):
                    if not trained:
                        raise ValueError('no trainable episodes under the selected supervision policy')
                    epoch += 1
                    cursor.update(episode_index=0, episode_state=None)
                    checkpoint('epoch_end')
                if stop['signal']:
                    checkpoint(stop['signal'])
                    print(json.dumps({'event': 'stopped', 'signal': stop['signal'],
                                      'window': trainer.windows_completed}), flush=True)
                    break
    finally:
        tracker.close(failed=sys.exc_info()[0] is not None)
    return trainer


def train(**kwargs):
    """Single-process by default; torchrun automatically enables data parallelism."""
    import os
    import torch.distributed as dist
    if int(os.environ.get('WORLD_SIZE', '1')) > 1 or (dist.is_initialized() and dist.get_world_size() > 1):
        if kwargs.get('model_parallel_devices') is not None:
            raise ValueError('model parallel training must launch with python, not torchrun')
        from .distributed_train import train_distributed
        return train_distributed(**kwargs)
    return _train(**kwargs)


def count_training_windows(records, config, *, limit_turns=None):
    """Windows cannot cross episodes; include every episode's incomplete tail."""
    if limit_turns is not None and limit_turns < 2:
        raise ValueError('limit_turns must be >= 2')
    total = 0
    for record in records:
        if config.action_supervision == 'success' and record.get('reward') != 1:
            continue
        count = len(load_episode(record).turns)
        if limit_turns is not None:
            count = min(count, limit_turns)
        total += math.ceil(count / config.unroll_steps)
    return total
