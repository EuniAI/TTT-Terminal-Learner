"""Offline architecture parity checks with locally generated HF checkpoints."""
import pytest
import torch

transformers = pytest.importorskip('transformers')
tokenizers = pytest.importorskip('tokenizers')

from terminal_learner.model.pretrained import import_pretrained
from terminal_learner.checkpoint import save_checkpoint, load_checkpoint
from terminal_learner.config import EncodingConfig
from terminal_learner.session import AgentSession
from terminal_learner.dataloader.tokenizer import Tokenizer, NATIVE_TOKENS


@pytest.mark.parametrize('requested_capacity', [None, 128])
@pytest.mark.parametrize('architecture,dtype', [
    ('qwen3', 'float32'), ('qwen3', 'bfloat16'),
    ('qwen2', 'float32'), ('llama', 'float32'),
])
def test_import_logits_cache_and_checkpoint(tmp_path, architecture, dtype, requested_capacity):
    torch.set_num_threads(1)
    torch.manual_seed(123)
    cls = {'qwen3': transformers.Qwen3Config,
           'qwen2': transformers.Qwen2Config,
           'llama': transformers.LlamaConfig}[architecture]
    # Same head dimension relation and GQA ratio as Qwen3-8B; tiny widths/layers.
    config = cls(vocab_size=288, hidden_size=32, intermediate_size=48,
                 num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=1,
                 head_dim=8, max_position_embeddings=8192, rope_theta=1000000.0,
                 attention_dropout=0.0, tie_word_embeddings=False)
    config._attn_implementation = 'eager'
    reference = transformers.AutoModelForCausalLM.from_config(config).to(getattr(torch, dtype)).eval()
    # Non-unit Q/K norms ensure the importer actually copies learned weights.
    if architecture == 'qwen3':
        with torch.no_grad():
            for layer in reference.model.layers:
                layer.self_attn.q_norm.weight.uniform_(0.5, 1.5)
                layer.self_attn.k_norm.weight.uniform_(0.5, 1.5)
    reference.save_pretrained(tmp_path)
    alphabet = sorted(tokenizers.pre_tokenizers.ByteLevel.alphabet())
    backend = tokenizers.Tokenizer(tokenizers.models.BPE(
        vocab={char: i for i, char in enumerate(alphabet)}, merges=[]))
    backend.pre_tokenizer = tokenizers.pre_tokenizers.ByteLevel(add_prefix_space=False)
    backend.decoder = tokenizers.decoders.ByteLevel()
    if architecture == 'qwen3':
        backend.add_special_tokens(list(NATIVE_TOKENS[:3]))
        backend.add_tokens(list(NATIVE_TOKENS[3:]))
        hf = transformers.PreTrainedTokenizerFast(tokenizer_object=backend,
            chat_template=Tokenizer.native_tiny().chat_template, eos_token='<|im_end|>', pad_token='<|endoftext|>')
    else:
        hf = transformers.PreTrainedTokenizerFast(tokenizer_object=backend)
    hf.save_pretrained(tmp_path)

    model, fast, tokenizer = import_pretrained(str(tmp_path), max_seq_len=requested_capacity, dtype=dtype)
    assert model.config.max_seq_len == (8192 if requested_capacity is None else requested_capacity)
    assert model.config.qk_norm == (architecture == 'qwen3')
    assert tokenizer.native == (architecture == 'qwen3')
    assert model.config.vocab_size == (288 if architecture == 'qwen3' else 293)
    ids = torch.tensor([[1, 17, 42, 71, 99]])
    tolerance = 0.01 if dtype == 'bfloat16' else 2e-5
    with torch.no_grad():
        actual = model(ids, fast)
        torch.testing.assert_close(actual[..., :288], reference(ids).logits, atol=tolerance, rtol=tolerance)
        _, cache = model.forward_cached(ids[:, :3], fast)
        cached, _ = model.forward_cached(ids[:, 3:], fast, cache)
        torch.testing.assert_close(cached[:, -1], actual[:, -1], atol=tolerance, rtol=tolerance)
    checkpoint = tmp_path / 'agent.pt'
    save_checkpoint(checkpoint, model, fast, tokenizer,
                    EncodingConfig(),
                    inner_lr=0.1, inner_clip=1.0)
    restored, restored_fast, encoder, _ = load_checkpoint(checkpoint)
    with torch.no_grad():
        torch.testing.assert_close(restored(ids, restored_fast), actual)
    assert restored.config.max_seq_len == model.config.max_seq_len
    session = AgentSession(restored, restored_fast, encoder)
    session.reset('list files', '$')
    theta = {k: v.detach().clone() for k, v in restored.named_parameters()}
    action = {'commands': [{'keystrokes': 'ls\n', 'duration': 0.1}], 'task_complete': False}
    if architecture == 'qwen3' and requested_capacity == 128:
        with pytest.raises(ValueError, match='model context capacity'):
            session.observe(action, 'app/')
        return
    session.observe(action, 'app/')
    assert all(torch.equal(theta[k], v) for k, v in restored.named_parameters())
    assert any(not torch.equal(restored_fast[k], v) for k, v in session.fast.items())
