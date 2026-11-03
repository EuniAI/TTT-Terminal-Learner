import hashlib
import io
import json
from pathlib import Path
from urllib.error import URLError

import pytest

from scripts.prepare_data import prepare, load_lock, DEFAULT_LOCK, REPO, SUBMISSION, download
from terminal_learner.dataloader.trajectory import read_manifest, load_episode


def fixture(tmp_path):
    records, contents = [], {}
    revision = 'a' * 40
    for i, reward in enumerate([1.0, 0.0, None]):
        trial = f'task__{i}'
        trajectory = {'agent': {'name': 'terminus-2', 'model_name': 'anthropic/claude-opus-4-6'},
            'steps': [{'source': 'user', 'message': 'Task Description:\nlist files'},
                {'source': 'agent', 'step_id': 2, 'reasoning_content': 'Inspect the files.',
                 'tool_calls': [{'function_name': 'bash_command', 'arguments': {'keystrokes': 'ls\n'}}],
                 'observation': {'results': [{'content': 'src'}]}}]}
        result = {'trial_name': trial, 'task_name': 'task',
                  'verifier_result': {'rewards': {'reward': reward}},
                  'exception_info': {'exception_type': 'TimeoutError'} if i == 2 else None}
        record = {'run': '2026-02-05', 'trial': trial, 'task': 'task'}
        for kind, value in [('trajectory', trajectory), ('result', result)]:
            content = json.dumps(value).encode()
            record[kind + '_sha256'] = hashlib.sha256(content).hexdigest()
            suffix = 'agent/trajectory.json' if kind == 'trajectory' else 'result.json'
            url = f'https://huggingface.co/datasets/{REPO}/resolve/{revision}/{SUBMISSION}/2026-02-05/{trial}/{suffix}'
            contents[url] = content
        records.append(record)
    lock = {'format': 'terminus2-opus46-data-lock-v1', 'repo_id': REPO, 'submission': SUBMISSION,
            'revision': revision, 'expected_count': 3, 'expected_tasks': 1, 'records': records}
    path = tmp_path / 'lock.json'
    path.write_text(json.dumps(lock))
    calls = []
    def opener(request, *, timeout):
        calls.append(request.full_url)
        return io.BytesIO(contents[request.full_url])
    return path, contents, calls, opener


def test_locked_collection_contains_exactly_the_selected_experts():
    lock = load_lock(DEFAULT_LOCK)
    assert len(lock['records']) == lock['expected_count'] == 445
    assert len({r['task'] for r in lock['records']}) == lock['expected_tasks'] == 89
    assert len(lock['revision']) == 40


def test_prepare_reuse_repair_and_read_only_verification(tmp_path):
    lock, _, calls, opener = fixture(tmp_path)
    output = tmp_path / 'moved-location'
    manifest, downloaded = prepare(output, lock_file=lock, opener=opener)
    assert downloaded == len(calls) == 6
    assert manifest['retained_count'] == 3
    assert [r['reward'] for r in manifest['records']] == [1., 0., None]
    assert manifest['summary']['with_exception'] == 1
    assert manifest['summary']['with_reasoning_content'] == 3
    records, fingerprint = read_manifest(output / 'expert-manifest.json')
    assert len(records) == 3
    assert all(load_episode(r).turns[0].reasoning_content == 'Inspect the files.' for r in records)
    first = (output / 'expert-manifest.json').read_bytes()
    _, downloaded = prepare(output, lock_file=lock, opener=opener)
    assert downloaded == 0 and len(calls) == 6
    assert (output / 'expert-manifest.json').read_bytes() == first
    path = output / manifest['records'][1]['trajectory_file']
    path.write_text('corrupted')
    with pytest.raises(ValueError, match='SHA-256'):
        prepare(output, lock_file=lock, opener=opener, verify_only=True)
    assert len(calls) == 6 and path.read_text() == 'corrupted'
    _, downloaded = prepare(output, lock_file=lock, opener=opener)
    assert downloaded == 1 and len(calls) == 7
    assert read_manifest(output / 'expert-manifest.json')[1] == fingerprint
    timestamp = (output / 'expert-manifest.json').stat().st_mtime_ns
    prepare(output, lock_file=lock, opener=opener, verify_only=True)
    assert (output / 'expert-manifest.json').stat().st_mtime_ns == timestamp
    assert len(calls) == 7


def test_failed_download_preserves_old_manifest_and_cleans_temp_files(tmp_path):
    lock, _, _, opener = fixture(tmp_path)
    output = tmp_path / 'data'
    manifest, _ = prepare(output, lock_file=lock, opener=opener)
    path = output / 'expert-manifest.json'
    before = path.read_bytes()
    broken = output / manifest['records'][0]['result_file']
    broken.write_text('old broken file')
    def bad(request, **kwargs):
        return io.BytesIO(b'wrong data')
    with pytest.raises(RuntimeError, match='SHA-256 mismatch'):
        prepare(output, lock_file=lock, opener=bad, retries=0)
    assert path.read_bytes() == before
    assert broken.read_text() == 'old broken file'
    assert not list(output.rglob('*.part'))
    assert not list(output.rglob('*.tmp'))


def test_retry_and_atomic_file_publication(tmp_path, monkeypatch):
    monkeypatch.setattr('scripts.prepare_data.time.sleep', lambda seconds: None)
    calls = []
    def flaky(request, **kwargs):
        calls.append(request.full_url)
        if len(calls) == 1:
            raise URLError('temporary connection error')
        return io.BytesIO(b'complete')
    path = tmp_path / 'download.json'
    assert download('https://huggingface.co/test', path, hashlib.sha256(b'complete').hexdigest(),
                    opener=flaky, retries=1)
    assert path.read_bytes() == b'complete' and len(calls) == 2


@pytest.mark.parametrize('field,value', [('revision', 'main'), ('repo_id', 'another/model'),
                                         ('expected_count', 444), ('expected_tasks', 90)])
def test_invalid_lock_rejected(tmp_path, field, value):
    path, _, _, _ = fixture(tmp_path)
    lock = json.loads(path.read_text())
    lock[field] = value
    path.write_text(json.dumps(lock))
    with pytest.raises(ValueError):
        load_lock(path)
