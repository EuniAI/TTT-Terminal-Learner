from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
import torch
import yaml

@dataclass
class ModelConfig:
    vocab_size: int = 261
    hidden_size: int = 32
    intermediate_size: int = 64
    num_layers: int = 2
    num_heads: int = 4
    num_kv_heads: int = 4
    adaptive_layers: int = 1
    max_seq_len: int = 4096
    rope_theta: float = 10000.0
    rms_norm_eps: float = 1e-6
    dtype: str = "float32"
    tie_embeddings: bool = False
    qkv_bias: bool = False
    qk_norm: bool = False  # Qwen3 per-head normalization; default preserves old checkpoints.
    attention_backend: str = "math"
    model_id: str = "tiny-random"
    revision: str = "native-v1"

    @property
    def torch_dtype(self):
        return getattr(torch, self.dtype)

    def validate(self):
        for name in ("vocab_size", "hidden_size", "intermediate_size", "num_layers",
                     "num_heads", "num_kv_heads", "max_seq_len"):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be positive")
        if not 1 <= self.adaptive_layers <= self.num_layers:
            raise ValueError("adaptive_layers must be between 1 and num_layers")
        if self.hidden_size % self.num_heads or self.num_heads % self.num_kv_heads:
            raise ValueError("incompatible attention head counts")
        if (self.hidden_size // self.num_heads) % 2:
            raise ValueError("RoPE head dimension must be even")
        if self.dtype not in ("float32", "float64", "bfloat16"):
            raise ValueError("use float32, float64, or bfloat16")
        if self.attention_backend not in ("math", "flashbob"):
            raise ValueError("attention_backend must be math or flashbob")


@dataclass
class EncodingConfig:
    """No text budgets: retain complete task, history, and targets."""

    enable_thinking: bool = False  # Old checkpoints keep their original non-thinking behavior.

    def validate(self, max_seq_len):
        if type(self.enable_thinking) is not bool:
            raise ValueError('enable_thinking must be a boolean')
        if max_seq_len is not None and max_seq_len < 2:
            raise ValueError('model context capacity must be >= 2')

    @classmethod
    def from_checkpoint(cls, values):
        # Old truncation settings no longer apply, including when resuming.
        legacy = {'task_tokens', 'history_tokens', 'loss_chunk_tokens', 'max_observation_tokens'}
        unknown = set(values) - legacy - {'enable_thinking'}
        if unknown:
            raise ValueError(f'unknown encoding settings: {sorted(unknown)}')
        return cls(enable_thinking=values.get('enable_thinking', False))

@dataclass
class TrainConfig:
    inner_lr: float = 0.1
    inner_clip: float = 1.0
    outer_lr: float = 1e-5
    outer_lr_schedule: str = "constant"
    outer_warmup_ratio: float = 0.1
    outer_end_lr: float = 1e-5
    weight_decay: float = 0.0
    unroll_steps: int = 2
    action_supervision: str = 'all'

    def validate(self):
        for key in ('inner_lr', 'inner_clip', 'outer_lr'):
            if not math.isfinite(getattr(self, key)) or getattr(self, key) <= 0:
                raise ValueError(f'{key} must be finite and positive')
        if not math.isfinite(self.weight_decay) or self.weight_decay < 0:
            raise ValueError('invalid weight decay')
        if type(self.unroll_steps) is not int or self.unroll_steps < 2:
            raise ValueError('unroll_steps must be >= 2 for observation -> subsequent action credit')
        if self.outer_lr_schedule not in ('constant', 'warmup_cosine'):
            raise ValueError('outer_lr_schedule must be constant or warmup_cosine')
        if not math.isfinite(self.outer_warmup_ratio) or not 0 <= self.outer_warmup_ratio < 1:
            raise ValueError('outer_warmup_ratio must be in [0, 1)')
        if not math.isfinite(self.outer_end_lr) or self.outer_end_lr < 0:
            raise ValueError('outer_end_lr must be finite and nonnegative')
        if self.outer_lr_schedule == 'warmup_cosine' and self.outer_end_lr > self.outer_lr:
            raise ValueError('outer_end_lr must not exceed outer_lr')
        if self.action_supervision not in ('all', 'success'):
            raise ValueError('action_supervision must be all or success')


@dataclass
class ModelSourceConfig:
    model_id: str = 'tiny-random'
    revision: str | None = None
    dtype: str = 'float32'
    attention_backend: str = 'math'
    adaptive_layers: int = 1
    max_seq_len: int | None = None  # None uses the pretrained model capacity.


@dataclass
class DataConfig:
    manifest: str = '../../data/terminus2/expert-manifest.json'
    test_fraction: float = 0.0


@dataclass
class RunSettings:
    device: str = 'cpu'
    model_parallel_devices: list[str] | None = None
    model_parallel_layer_counts: list[int] | None = None
    distributed_timeout_seconds: int = 1800
    distributed_bucket_mb: int = 25
    output: str = '../../runs/agent.pt'
    keep_checkpoints: int = 3
    save_every_windows: int = 50
    wandb_mode: str = "disabled"
    wandb_project: str | None = None
    wandb_name: str | None = None
    wandb_entity: str | None = None
    wandb_dir: str | None = None
    wandb_new_run: bool = False
    epochs: int = 1
    seed: int = 0
    limit_episodes: int | None = None
    limit_turns: int | None = None


def _typed_section(cls, values, section):
    """Reject YAML typos and implicit strings/bools where numbers are required."""
    from types import UnionType
    from typing import get_args, get_origin, get_type_hints
    if not isinstance(values, dict):
        raise ValueError(f'{section} must be a mapping')
    hints = get_type_hints(cls)
    unknown = set(values) - set(hints)
    if unknown:
        raise ValueError(f'unknown {section} keys: {sorted(unknown)}')
    for key, value in values.items():
        hint = hints[key]
        types = get_args(hint) if get_origin(hint) is UnionType else (hint,)
        valid = any((type(value) is list and all(type(v) is get_args(t)[0] for v in value))
                    if get_origin(t) is list else (type(value) is t or (t is float and type(value) is int))
                    for t in types)
        if not valid:
            raise ValueError(f'invalid type for {section}.{key}: expected {hint}')
    return cls(**values)


@dataclass
class RunConfig:
    model: ModelSourceConfig = field(default_factory=ModelSourceConfig)
    training: TrainConfig = field(default_factory=TrainConfig)
    encoding: EncodingConfig = field(default_factory=EncodingConfig)
    data: DataConfig = field(default_factory=DataConfig)
    run: RunSettings = field(default_factory=RunSettings)

    @classmethod
    def load(cls, path):
        path = Path(path).resolve()
        values = yaml.safe_load(path.read_text())
        if not isinstance(values, dict):
            raise ValueError('experiment YAML must be a mapping')
        sections = {'model': ModelSourceConfig, 'training': TrainConfig,
                    'encoding': EncodingConfig, 'data': DataConfig, 'run': RunSettings}
        unknown = set(values) - set(sections)
        if unknown:
            raise ValueError(f'unknown config sections: {sorted(unknown)}')
        config = cls(**{name: _typed_section(kind, values.get(name, {}), name)
                        for name, kind in sections.items()})
        # File paths in YAML are relative to that YAML file, independent of cwd.
        for obj, key in ((config.data, 'manifest'), (config.run, 'output')):
            value = Path(getattr(obj, key)).expanduser()
            setattr(obj, key, str((path.parent / value).resolve()))
        if config.run.wandb_dir is not None:
            config.run.wandb_dir = str((path.parent / Path(config.run.wandb_dir).expanduser()).resolve())
        config.validate()
        return config

    def validate(self):
        # Also check types after programmatic / command-line overrides.
        for item in fields(self):
            value = getattr(self, item.name)
            _typed_section(type(value), asdict(value), item.name)
        self.training.validate()
        capacity = self.model.max_seq_len
        if capacity is None and self.model.model_id == 'tiny-random':
            capacity = ModelConfig().max_seq_len
        self.encoding.validate(capacity)
        if self.model.attention_backend not in ('math', 'flashbob'):
            raise ValueError('model.attention_backend must be math or flashbob')
        if self.model.dtype not in ('float32', 'float64', 'bfloat16'):
            raise ValueError('model.dtype must be float32, float64 or bfloat16')
        if not self.model.model_id or self.model.adaptive_layers < 1 or (self.model.max_seq_len is not None and self.model.max_seq_len < 2):
            raise ValueError('invalid model source or dimensions')
        if self.model.model_id == 'tiny-random' and self.model.adaptive_layers > 2:
            raise ValueError('tiny model has only two layers')
        if not 0 <= self.data.test_fraction < 1:
            raise ValueError('data.test_fraction must be in [0, 1)')
        if not self.data.manifest or not self.run.output or not self.run.device:
            raise ValueError('manifest, output and device must not be empty')
        if self.run.model_parallel_devices is not None:
            devices = self.run.model_parallel_devices
            if not devices or len(set(devices)) != len(devices) or any(not d.startswith('cuda:') or not d[5:].isdigit() for d in devices):
                raise ValueError('model_parallel_devices must be distinct explicit CUDA devices')
        if self.run.model_parallel_layer_counts is not None:
            counts = self.run.model_parallel_layer_counts
            if self.run.model_parallel_devices is None or len(counts) != len(self.run.model_parallel_devices) or any(n < 1 for n in counts):
                raise ValueError('model_parallel_layer_counts must match devices and be positive')
        if self.run.save_every_windows < 1:
            raise ValueError('run.save_every_windows must be positive')
        if self.run.distributed_timeout_seconds < 1 or self.run.distributed_bucket_mb < 1:
            raise ValueError('distributed timeout and bucket size must be positive')
        if self.run.wandb_mode not in ('disabled', 'offline', 'online'):
            raise ValueError('invalid run.wandb_mode')
        if self.run.wandb_mode != 'disabled' and not self.run.wandb_project:
            raise ValueError('run.wandb_project required for W&B')
        if self.run.wandb_new_run and self.run.wandb_mode == 'disabled':
            raise ValueError('wandb_new_run requires enabled W&B tracking')
        if self.run.wandb_dir is not None and not self.run.wandb_dir.strip():
            raise ValueError('wandb_dir must not be empty')
        if self.run.keep_checkpoints < 1:
            raise ValueError('run.keep_checkpoints must be positive')
        if self.run.epochs < 1:
            raise ValueError('run.epochs must be positive')
        for name, minimum in (('limit_episodes', 1), ('limit_turns', 2)):
            value = getattr(self.run, name)
            if value is not None and value < minimum:
                raise ValueError(f'run.{name} must be >= {minimum} or null')
