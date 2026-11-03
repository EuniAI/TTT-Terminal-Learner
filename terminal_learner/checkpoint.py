from __future__ import annotations

import os
import tempfile
import uuid
from dataclasses import asdict
from pathlib import Path
import torch
from .config import ModelConfig, EncodingConfig
from .model.transformer import TerminalTransformer
from .model.backends import resolve_backend
from .model.meta_model import copy_fast
from .model.numerics import validate_finite_tree
from .dataloader.tokenizer import Tokenizer
from .dataloader.encoding import Encoder

def save_checkpoint(path, model, initial_fast, tokenizer, encoder_config, *, inner_lr,
                    inner_clip, training=None, keep=3):
    if type(keep) is not int or keep < 1:
        raise ValueError("checkpoint retention must be a positive integer")
    model.validate_fast(initial_fast)
    validate_finite_tree(model.state_dict(), 'checkpoint.theta')
    validate_finite_tree(initial_fast, 'checkpoint.initial_fast')
    validate_finite_tree(training, 'checkpoint.training')
    payload = {'format': 'ttt-terminal-agent-v1', 'model_config': asdict(model.config),
               'theta': {k: v.detach().cpu() for k, v in model.state_dict().items()},
               'initial_fast': {k: v.detach().cpu() for k, v in initial_fast.items()},
               'tokenizer': tokenizer.state_dict(), 'encoding_config': asdict(encoder_config),
               'inner_lr': inner_lr, 'inner_clip': inner_clip, 'training': training or {}}
    if training and 'windows_completed' in training:
        retained_atomic_save(payload, path, window=training['windows_completed'], keep=keep)
    else:
        atomic_save(payload, path)  # Standalone inference export has no training cursor.


def _sync_directory(path):
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def atomic_save(payload, path):
    """Write and flush a unique temporary file before atomically publishing it."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=path.name + '.', suffix='.tmp', dir=path.parent)
    try:
        with os.fdopen(descriptor, 'wb') as output:
            torch.save(payload, output)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, path)
        _sync_directory(path.parent)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def retained_atomic_save(payload, path, *, window, keep=3):
    """Publish latest first, then retain the most recently saved window snapshots.

    Re-saving the same window replaces its snapshot. Latest is a hard link, so it
    does not duplicate the checkpoint's storage. Use one writer per output path.
    """
    if type(keep) is not int or keep < 1:
        raise ValueError('checkpoint retention must be a positive integer')
    if type(window) is not int or window < 0:
        raise ValueError('checkpoint window must be a nonnegative integer')
    path = Path(path)
    history = path.parent / (path.stem + '.checkpoints')
    snapshot = history / f'window-{window:012d}.pt'
    atomic_save(payload, snapshot)
    temporary = path.parent / f'.{path.name}.{uuid.uuid4().hex}.link'
    try:
        os.link(snapshot, temporary)
        os.replace(temporary, path)
        _sync_directory(path.parent)
    finally:
        if temporary.exists():
            temporary.unlink()
    # Prune only after the latest pointer is durable. Time order permits resuming
    # an older checkpoint without immediately deleting the newly saved result.
    candidates = [p for p in history.iterdir()
                  if p.is_file() and not p.is_symlink() and p.name.startswith('window-')
                  and p.suffix == '.pt' and p.stem.removeprefix('window-').isdigit()]
    older = sorted((p for p in candidates if p != snapshot),
                   key=lambda p: p.stat().st_mtime_ns, reverse=True)
    for obsolete in older[keep - 1:]:
        obsolete.unlink()
    _sync_directory(history)


def load_checkpoint(path, device='cpu', *, attention_backend=None, devices=None, layer_counts=None):
    payload = torch.load(path, map_location='cpu', weights_only=True)
    if payload.get('format') != 'ttt-terminal-agent-v1':
        raise ValueError('not a TTT terminal agent checkpoint')
    validate_finite_tree(payload, 'checkpoint')
    config = ModelConfig(**payload['model_config'])
    if attention_backend is not None:
        config.attention_backend = attention_backend
    from .model.placement import place_model
    resolve_backend(config, devices[0] if devices else device)
    model = place_model(TerminalTransformer(config), device, devices=devices, layer_counts=layer_counts)
    model.configure_backend()
    model.load_state_dict(payload['theta'], strict=True)
    fast = model.place_fast(payload['initial_fast'])
    model.validate_fast(fast)
    tokenizer = Tokenizer(**payload['tokenizer'])
    if tokenizer.vocab_size != model.config.vocab_size:
        raise ValueError('tokenizer/model vocabulary mismatch')
    encoder = Encoder(tokenizer, EncodingConfig.from_checkpoint(payload['encoding_config']), model.config.max_seq_len)
    return model, fast, encoder, payload
