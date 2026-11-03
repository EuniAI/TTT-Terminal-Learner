import pytest
import torch

from terminal_learner.checkpoint import save_checkpoint, load_checkpoint
from terminal_learner.config import ModelConfig, EncodingConfig
from terminal_learner.dataloader.tokenizer import Tokenizer
from terminal_learner.model.transformer import create_model
from terminal_learner.model.optimizer import SlowWeightAdamW


@pytest.mark.parametrize('part', ['theta', 'initial_fast', 'training'])
def test_bad_save_preserves_previous_checkpoint(tmp_path, part):
    model, fast = create_model(ModelConfig(hidden_size=8, intermediate_size=12,
        num_layers=1, num_heads=2, num_kv_heads=1))
    path = tmp_path / 'model.pt'
    def save(training=None):
        save_checkpoint(path, model, fast, Tokenizer(), EncodingConfig(),
                        inner_lr=.1, inner_clip=1, training=training)
    save()
    original = path.read_bytes()
    training = None
    with torch.no_grad():
        if part == 'theta':
            next(model.parameters()).view(-1)[0] = float('nan')
        elif part == 'initial_fast':
            next(iter(fast.values())).view(-1)[0] = float('inf')
        else:
            training = {'progress': {'fast': torch.tensor(float('nan'))}}
    with pytest.raises(FloatingPointError, match=part):
        save(training)
    assert path.read_bytes() == original
    # Corrupted on-disk payloads must also be refused before model construction.
    payload = torch.load(path, weights_only=True)
    payload[part]['corrupt'] = torch.tensor(float('inf'))
    torch.save(payload, path)
    with pytest.raises(FloatingPointError, match=part):
        load_checkpoint(path)


@pytest.mark.parametrize('part', ['master', 'exp_avg', 'exp_avg_sq'])
def test_optimizer_rejects_corrupt_state_and_updates(part):
    p = torch.nn.Parameter(torch.ones(1, dtype=torch.bfloat16))
    optimizer = SlowWeightAdamW([p], lr=.001)
    optimizer.set_gradients([torch.ones_like(p)])
    optimizer.step()
    saved = optimizer.state_dict()
    if part == 'master':
        saved['fp32_master_weights']['0'][0] = float('nan')
    else:
        saved['state'][0][part][0] = float('inf')
    with pytest.raises(FloatingPointError):
        optimizer.load_state_dict(saved)
    # A failed step must not copy bad masters into the BF16 forward model.
    before = p.detach().clone()
    optimizer.master_parameters[0].data[0] = float('nan')
    with pytest.raises(FloatingPointError):
        optimizer.step()
    assert torch.equal(before, p)
