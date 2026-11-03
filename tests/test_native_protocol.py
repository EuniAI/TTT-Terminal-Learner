"""Qwen3 chat-template/token parity, label alignment, inference, and recovery."""
import json

import pytest
import torch
from tokenizers import Tokenizer as Backend, models, pre_tokenizers, decoders
from transformers import PreTrainedTokenizerFast

from terminal_learner.config import EncodingConfig, ModelConfig, TrainConfig
from terminal_learner.dataloader.encoding import Encoder
from terminal_learner.dataloader.tokenizer import Tokenizer, NATIVE_TOKENS
from terminal_learner.protocol import TERMINAL_TOOL, canonical_action, parse_tool_action
from terminal_learner.model.transformer import create_model
from terminal_learner.model.meta_model import observation_loss, copy_fast, inner_update, window_loss
from terminal_learner.model.data import Episode, Turn
from terminal_learner.session import AgentSession
from terminal_learner.checkpoint import save_checkpoint, load_checkpoint


ACTION = {'commands': [{'keystrokes': 'ls\n', 'duration': 1.0}], 'task_complete': False}


@pytest.fixture
def native():
    alphabet = sorted(pre_tokenizers.ByteLevel.alphabet())
    vocab = {char: i for i, char in enumerate(alphabet)}
    vocab['ĊĊ'] = len(vocab)
    backend = Backend(models.BPE(vocab=vocab, merges=[('Ċ', 'Ċ')]))
    backend.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    backend.decoder = decoders.ByteLevel()
    backend.add_special_tokens(list(NATIVE_TOKENS[:3]))
    backend.add_tokens(list(NATIVE_TOKENS[3:]))
    hf = PreTrainedTokenizerFast(tokenizer_object=backend, eos_token='<|im_end|>',
        pad_token='<|endoftext|>', chat_template=Tokenizer.native_tiny().chat_template)
    tokenizer = Tokenizer(backend.to_str(), 288, protocol='qwen3', chat_template=hf.chat_template)
    encoder = Encoder(tokenizer, EncodingConfig(), 4096)
    return hf, tokenizer, encoder


def messages(task, history):
    return [{'role': 'system', 'content': 'You are a terminal agent. Use the terminal tool to complete the task.'},
            {'role': 'user', 'content': task}] + history


def assistant(action):
    return {'role': 'assistant', 'content': '', 'tool_calls': [
        {'type': 'function', 'function': {'name': 'terminal', 'arguments': action}}]}


@pytest.mark.parametrize('output', ['README src', '', '\n\n中文\n',
    '<tool_response>literal</tool_response>\n<|im_end|>', '__TTT_TOOL_CONTENT_BOUNDARY_7b4f3d__'])
def test_exact_hf_template_tokens_and_boundary(native, output):
    hf, tok, enc = native
    history = enc.initial_history('$')
    base = messages('list files', history)
    def official(msg, generation=False):
        return hf.apply_chat_template(msg, tools=[TERMINAL_TOOL], tokenize=True,
                                      enable_thinking=False, add_generation_prompt=generation)
    assert enc.prompt('list files', history) == official(base, True)
    p, t = enc.action_example('list files', history, canonical_action(ACTION))
    assert p + t == official(base + [assistant(ACTION)])
    assert tok.decode(t).startswith('<tool_call>')
    assert '<think>\n\n</think>' in tok.decode(p)
    p, t = enc.observation_example('list files', history, canonical_action(ACTION), output)
    assert p + t == official(base + [assistant(ACTION), {'role': 'tool', 'content': output}])
    if output.startswith('\n'):
        assert len(p) < len(enc.prompt('list files', history, action=canonical_action(ACTION)))
    history = enc.history(history, canonical_action(ACTION), output)
    assert enc.prompt('list files', history) == official(base + [assistant(ACTION),
                                                             {'role': 'tool', 'content': output}], True)
    assert tok.vocab_size == 288
    assert tok.action_end_id == hf.eos_token_id


def test_native_session_cache_update_and_checkpoint(native, tmp_path, monkeypatch):
    hf, tok, enc = native
    torch.set_num_threads(1)
    torch.manual_seed(5)
    model, fast = create_model(ModelConfig(vocab_size=288, hidden_size=4, intermediate_size=6,
        num_layers=2, adaptive_layers=1, num_heads=1, num_kv_heads=1, dtype='float64', max_seq_len=4096))
    session = AgentSession(model, fast, enc)
    session.reset('list files', '$')
    theta = {k: v.clone() for k, v in model.named_parameters()}
    generated = '<tool_call>\n' + json.dumps({'name': 'terminal', 'arguments': ACTION}) + '\n</tool_call>'
    tokens = iter(tok.encode(generated) + [tok.action_end_id])
    def scripted(ids, phi, cache=None):
        logits = torch.full((1, 1, tok.vocab_size), -100., dtype=torch.float64)
        logits[0, 0, next(tokens)] = 100
        return logits, []
    monkeypatch.setattr(session, '_action_forward', scripted)
    result = session.act()
    assert result['valid'] and result['action'] == ACTION
    assert result['raw'] == generated  # Includes native wrappers, no custom markers.
    for output in ['src', '\n\ntests']:
        reference = copy_fast(session.fast)
        loss, count, _ = observation_loss(model, reference, enc, session.task, session.history,
                                          canonical_action(ACTION), output)
        updated = inner_update(reference, loss, lr=.1, clip_norm=1, create_graph=False)
        observed = session.observe(generated, output)
        assert observed['tokens'] == count
        assert observed['observation_nll'] == pytest.approx(loss.item(), abs=1e-10)
        for key in updated:
            torch.testing.assert_close(session.fast[key], updated[key])
    assert all(torch.equal(p, theta[k]) for k, p in model.named_parameters())
    # Native template/IDs survive checkpoint and offline restore, without HF downloads.
    path = tmp_path / 'native.pt'
    save_checkpoint(path, model, fast, tok, EncodingConfig(), inner_lr=.1, inner_clip=1)
    loaded, initial, encoder, payload = load_checkpoint(path)
    assert encoder.tok.native and encoder.tok.chat_template == hf.chat_template
    assert encoder.prompt('list files', session.history) == enc.prompt('list files', session.history)
    assert loaded.config.vocab_size == 288
    assert payload['tokenizer']['protocol'] == 'qwen3'


@pytest.mark.parametrize('thinking', [False, True])
def test_native_two_round_meta_gradient(native, thinking):
    _, tok, enc = native
    enc = Encoder(tok, EncodingConfig(enable_thinking=thinking), 4096)
    torch.set_num_threads(1)
    torch.manual_seed(12)
    model, fast = create_model(ModelConfig(vocab_size=tok.vocab_size, hidden_size=4, intermediate_size=6,
        num_layers=1, adaptive_layers=1, num_heads=1, num_kv_heads=1, dtype='float64', max_seq_len=4096))
    action = canonical_action(ACTION)
    episode = Episode('task', '$', [Turn(action, 'src', 1, 'Inspect files.'),
                                   Turn(action, 'done', 2, 'Run the next command.')], 'trial', 'task', 0)
    parameter = model.layers[0].self_attn.q_proj.weight
    def objective(differentiable):
        loss, _, _, _ = window_loss(model, copy_fast(fast), enc, episode, episode.turns,
            enc.initial_history('$'), TrainConfig(), differentiable=differentiable)
        return torch.autograd.grad(loss, parameter)[0]
    exact, detached = objective(True), objective(False)
    assert torch.isfinite(exact).all()
    assert (exact - detached).abs().max().item() > 1e-10


@pytest.mark.parametrize('text', [
    '{"commands":[],"task_complete":true}',
    '<tool_call>{"name":"other","arguments":{}}</tool_call>',
    '<tool_call>{"name":"terminal","arguments":{}}</tool_call>',
    '<tool_call>{"name":"terminal","arguments":{}}',
])
def test_invalid_native_action_rejected(text):
    with pytest.raises((ValueError, TypeError)):
        parse_tool_action(text)


def test_old_checkpoint_without_protocol_metadata_keeps_legacy(tmp_path):
    tok = Tokenizer()
    model, fast = create_model(ModelConfig())
    path = tmp_path / 'legacy.pt'
    save_checkpoint(path, model, fast, tok, EncodingConfig(), inner_lr=.1, inner_clip=1)
    payload = torch.load(path, weights_only=True)
    payload['tokenizer'].pop('protocol')
    payload['tokenizer'].pop('chat_template')
    torch.save(payload, path)
    _, _, enc, _ = load_checkpoint(path)
    assert not enc.tok.native
    assert enc.tok.action_end_id == tok.ids['action_end']


def test_thinking_template_supervision_and_observation_boundary(native):
    hf, tok, _ = native
    enc = Encoder(tok, EncodingConfig(enable_thinking=True), 4096)
    reasoning = '先查看目录，再决定如何运行测试。'
    history = enc.initial_history('$')
    base = messages('list files', history)
    assistant_message = {**assistant(ACTION), 'reasoning_content': reasoning}
    prompt, targets = enc.action_example('list files', history, canonical_action(ACTION),
                                        reasoning_content=reasoning)
    assert prompt == hf.apply_chat_template(base, tools=[TERMINAL_TOOL],
        add_generation_prompt=True, enable_thinking=True)
    assert reasoning not in tok.decode(prompt)
    assert tok.decode(targets).startswith('<think>\n' + reasoning + '\n</think>')
    assert prompt + targets == hf.apply_chat_template(base + [assistant_message],
        tools=[TERMINAL_TOOL], enable_thinking=True)
    prompt, targets = enc.observation_example('list files', history, canonical_action(ACTION),
                                             'src tests', reasoning_content=reasoning)
    assert reasoning in tok.decode(prompt)
    assert reasoning not in tok.decode(targets)
    assert prompt + targets == hf.apply_chat_template(base + [assistant_message,
        {'role': 'tool', 'content': 'src tests'}], tools=[TERMINAL_TOOL], enable_thinking=True)
    history = enc.history(history, canonical_action(ACTION), 'src tests', reasoning_content=reasoning)
    assert history[-2]['reasoning_content'] == reasoning
    assert enc.prompt('list files', history) == hf.apply_chat_template(base + [assistant_message,
        {'role': 'tool', 'content': 'src tests'}], tools=[TERMINAL_TOOL],
        add_generation_prompt=True, enable_thinking=True)


@pytest.mark.parametrize('mode', ['pending', 'raw', 'explicit'])
def test_generated_reasoning_reaches_observe_history_and_checkpoint(native, tmp_path, monkeypatch, mode):
    _, tok, _ = native
    enc = Encoder(tok, EncodingConfig(enable_thinking=True), 4096)
    torch.set_num_threads(1)
    model, fast = create_model(ModelConfig(vocab_size=288, hidden_size=4, intermediate_size=6,
        num_layers=1, adaptive_layers=1, num_heads=1, num_kv_heads=1, max_seq_len=4096, dtype='float64'))
    session = AgentSession(model, fast, enc)
    session.reset('task', '$')
    reasoning = 'Inspect files before choosing the next command.'
    raw = '<think>\n' + reasoning + '\n</think>\n\n<tool_call>\n' + json.dumps(
        {'name': 'terminal', 'arguments': ACTION}) + '\n</tool_call>'
    iterator = iter(tok.encode(raw) + [tok.action_end_id])
    def generate(ids, phi, cache=None):
        logits = torch.full((1, 1, tok.vocab_size), -100.)
        logits[0, 0, next(iterator)] = 100
        return logits, []
    monkeypatch.setattr(session, '_action_forward', generate)
    result = session.act()
    assert result['reasoning_content'] == reasoning and result['action'] == ACTION
    reference = copy_fast(fast)
    loss, _, _ = observation_loss(model, reference, enc, session.task, session.history,
        canonical_action(ACTION), 'files', reasoning_content=reasoning)
    expected = inner_update(reference, loss, lr=.1, clip_norm=1, create_graph=False)
    if mode == 'raw':
        session._pending_action = None
        session.observe(result['raw'], 'files')
    elif mode == 'explicit':
        session._pending_action = None
        session.observe(result['action'], 'files', reasoning_content=reasoning)
    else:
        session.observe(result['action'], 'files')
    assert session.history[-2]['reasoning_content'] == reasoning
    assert session._pending_action is None
    for key in expected:
        torch.testing.assert_close(session.fast[key], expected[key])
    path = tmp_path / 'thinking.pt'
    save_checkpoint(path, model, fast, tok, enc.config, inner_lr=.1, inner_clip=1)
    restored = AgentSession.from_checkpoint(path)
    assert restored.encoder.config.enable_thinking is True
    session.reset('another task')
    assert session._pending_action is None


def test_recorded_expert_reasoning_is_loaded(tmp_path):
    from terminal_learner.dataloader.trajectory import load_episode
    path = tmp_path / 'trajectory.json'
    path.write_text(json.dumps({'agent': {'name': 'terminus-2', 'model_name': 'anthropic/claude-opus-4-6'},
        'steps': [{'source': 'user', 'message': 'Task Description:\nlist files'},
            {'source': 'agent', 'step_id': 2, 'reasoning_content': 'Inspect the directory first.',
             'message': 'Analysis: separate presentation text',
             'tool_calls': [{'function_name': 'bash_command', 'arguments': {'keystrokes': 'ls\n'}}],
             'observation': {'results': [{'content': 'src'}]}}]}))
    episode = load_episode({'trajectory_file': str(path), 'trial': 'trial', 'task': 'task'})
    assert episode.turns[0].reasoning_content == 'Inspect the directory first.'
    assert 'Analysis:' not in episode.turns[0].reasoning_content
