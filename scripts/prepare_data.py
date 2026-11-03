#!/usr/bin/env python3
"""Prepare the locked 445 Terminus 2 / Claude Opus 4.6 expert trajectories.

Standard library only. No git clone, HF token, model download, or torch required.
"""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
from functools import lru_cache
import json
import math
import os
from pathlib import Path
import re
import sys
import ssl
import tempfile
import time
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LOCK = ROOT / 'configs/data/opus46.lock.json'
REPO = 'harborframework/terminal-bench-2-leaderboard'
SUBMISSION = 'submissions/terminal-bench/2.0/Terminus2__Claude-Opus-4.6'


@lru_cache(maxsize=1)
def tls_context():
    context = ssl.create_default_context()
    # Python.org macOS installs may lack their private CA file. Add the OS trust
    # bundle rather than disabling certificate verification or requiring pip.
    if sys.platform == 'darwin' and Path('/etc/ssl/cert.pem').is_file():
        context.load_verify_locations('/etc/ssl/cert.pem')
    return context


def open_url(request, *, timeout):
    return urlopen(request, timeout=timeout, context=tls_context())


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def load_lock(path):
    lock = json.loads(Path(path).read_text(encoding='utf-8'))
    if (lock.get('format') != 'terminus2-opus46-data-lock-v1'
            or lock.get('repo_id') != REPO or lock.get('submission') != SUBMISSION
            or not re.fullmatch(r'[0-9a-f]{40}', lock.get('revision', ''))):
        raise ValueError('invalid Opus 4.6 data lock or unpinned source')
    rows = lock['records']
    if len(rows) != lock['expected_count'] or len({r['trial'] for r in rows}) != len(rows):
        raise ValueError('data lock count mismatch or duplicate trial')
    if len({r['task'] for r in rows}) != lock['expected_tasks']:
        raise ValueError('data lock task count mismatch')
    for row in rows:
        for key in ('run', 'trial', 'task'):
            if row[key] in ('.', '..') or not re.fullmatch(r'[A-Za-z0-9_.-]+', row[key]):
                raise ValueError(f'invalid {key} in data lock')
        for key in ('trajectory_sha256', 'result_sha256'):
            if not re.fullmatch(r'[0-9a-f]{64}', row[key]):
                raise ValueError(f'invalid {key} in data lock')
    return lock


def download(url, destination, expected, *, timeout=60, retries=3, verify_only=False, opener=open_url):
    """Reuse a verified file, otherwise atomically replace it after a valid download."""
    destination = Path(destination)
    if destination.is_file() and sha256(destination) == expected:
        return False
    if verify_only:
        raise ValueError(f'missing or SHA-256 mismatch: {destination}')
    destination.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(retries + 1):
        descriptor, temporary = tempfile.mkstemp(prefix=destination.name + '.', suffix='.part', dir=destination.parent)
        try:
            digest = hashlib.sha256()
            with os.fdopen(descriptor, 'wb') as output:
                request = Request(url, headers={'User-Agent': 'TTT-Terminal-Learner-data-preparation/1'})
                with opener(request, timeout=timeout) as response:
                    for chunk in iter(lambda: response.read(1024 * 1024), b''):
                        output.write(chunk)
                        digest.update(chunk)
                output.flush()
                os.fsync(output.fileno())
            if digest.hexdigest() != expected:
                raise ValueError(f'download SHA-256 mismatch: {destination.name}')
            os.replace(temporary, destination)
            return True
        except (OSError, URLError, ValueError) as error:
            if attempt == retries or (isinstance(error, HTTPError) and error.code in (401, 403, 404)):
                raise RuntimeError(f'could not prepare {destination} from {url}: {error}') from error
            time.sleep(min(2 ** attempt, 8))
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)


def prepare_record(lock, record, output_dir, *, verify_only=False, timeout=60, retries=3, opener=open_url):
    base = f'https://huggingface.co/datasets/{REPO}/resolve/{lock["revision"]}/{SUBMISSION}/{record["run"]}/{record["trial"]}'
    directory = Path('opus46') / record['trial']
    urls = {'trajectory': base + '/agent/trajectory.json', 'result': base + '/result.json'}
    files, downloaded = {}, 0
    for kind in ('trajectory', 'result'):
        relative = directory / f'{kind}.json'
        downloaded += download(urls[kind], output_dir / relative, record[kind + '_sha256'],
                               timeout=timeout, retries=retries, verify_only=verify_only, opener=opener)
        files[kind] = relative
    trajectory = json.loads((output_dir / files['trajectory']).read_text(encoding='utf-8'))
    result = json.loads((output_dir / files['result']).read_text(encoding='utf-8'))
    agent = trajectory.get('agent') or {}
    if agent.get('name') != 'terminus-2' or agent.get('model_name') != 'anthropic/claude-opus-4-6':
        raise ValueError(f'non-expert source in {record["trial"]}')
    if result.get('trial_name') != record['trial'] or result.get('task_name') != record['task']:
        raise ValueError(f'result identity mismatch: {record["trial"]}')
    steps = trajectory.get('steps')
    if not isinstance(steps, list) or not any(s.get('source') == 'user' and isinstance(s.get('message'), str) for s in steps):
        raise ValueError(f'invalid trajectory steps: {record["trial"]}')
    reward = ((result.get('verifier_result') or {}).get('rewards') or {}).get('reward')
    if reward is not None and (type(reward) not in (int, float) or not math.isfinite(reward)):
        raise ValueError(f'invalid reward in {record["trial"]}')
    exception = (result.get('exception_info') or {}).get('exception_type')
    row = {'trial': record['trial'], 'task': record['task'],
           'trajectory_url': urls['trajectory'], 'result_url': urls['result'],
           'trajectory_file': files['trajectory'].as_posix(), 'result_file': files['result'].as_posix(),
           'trajectory_sha256': record['trajectory_sha256'], 'result_sha256': record['result_sha256'],
           'steps': len(steps), 'reward': reward, 'exception_type': exception, 'retained': True}
    tool_steps = [s for s in steps if s.get('source') == 'agent' and s.get('tool_calls')]
    reasoning = sum(isinstance(s.get('reasoning_content'), str) and bool(s['reasoning_content'].strip()) for s in tool_steps)
    return row, downloaded, len(tool_steps), reasoning


def atomic_json(path, value):
    descriptor, temporary = tempfile.mkstemp(prefix=path.name + '.', suffix='.tmp', dir=path.parent)
    try:
        with os.fdopen(descriptor, 'w', encoding='utf-8') as output:
            json.dump(value, output, ensure_ascii=False, indent=2, allow_nan=False)
            output.write('\n')
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def prepare(output_dir, *, lock_file=DEFAULT_LOCK, workers=4, timeout=60, retries=3,
            verify_only=False, opener=open_url, progress=None):
    if workers < 1 or not math.isfinite(timeout) or timeout <= 0 or retries < 0:
        raise ValueError('workers/timeout must be positive and retries must be nonnegative')
    output_dir = Path(output_dir).expanduser().resolve()
    lock = load_lock(lock_file)
    rows, downloaded, tool_steps, reasoning = [], 0, 0, 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(prepare_record, lock, record, output_dir, verify_only=verify_only,
            timeout=timeout, retries=retries, opener=opener) for record in lock['records']]
        try:
            for future in as_completed(futures):
                row, count, turns, reasons = future.result()
                rows.append(row)
                downloaded += count
                tool_steps += turns
                reasoning += reasons
                if progress and (len(rows) % 25 == 0 or len(rows) == len(futures)):
                    progress({'event': 'progress', 'verified_trajectories': len(rows),
                              'total': len(futures), 'downloaded_files': downloaded})
        except BaseException:
            for future in futures:
                future.cancel()
            raise  # Never publish a partial manifest.
    rows.sort(key=lambda r: r['trial'])
    summary = Counter({'trajectory_files': len(rows), 'result_files': len(rows),
                       'with_exception': sum(r['exception_type'] is not None for r in rows)})
    for row in rows:
        summary[f'reward_{row["reward"]}'] += 1
    summary.update(tool_steps=tool_steps, with_reasoning_content=reasoning)
    manifest = {'agent': 'terminus-2', 'teacher': 'anthropic/claude-opus-4-6',
        'source': f'https://huggingface.co/datasets/{REPO}/tree/{lock["revision"]}/{SUBMISSION}',
        'source_revision': lock['revision'], 'retention_policy': 'Keep all runs regardless of reward or exception.',
        'retained_count': len(rows), 'unique_tasks': len({r['task'] for r in rows}),
        'summary': dict(summary), 'training_supervision_policy': 'all; commands and recorded reasoning, including failed runs',
        'split_status': 'All records are training data; no holdout split.',
        'records': sorted(rows, key=lambda r: r['trial'])}
    if not verify_only:
        output_dir.mkdir(parents=True, exist_ok=True)
        atomic_json(output_dir / 'expert-manifest.json', manifest)
    return manifest, downloaded


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'data/terminus2')
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--timeout', type=float, default=60)
    parser.add_argument('--retries', type=int, default=3)
    parser.add_argument('--verify-only', action='store_true', help='offline read-only verification; no downloads or writes')
    args = parser.parse_args(argv)
    emit = lambda value: print(json.dumps(value, ensure_ascii=False), flush=True)
    try:
        manifest, downloaded = prepare(**vars(args), progress=emit)
    except (OSError, ValueError, RuntimeError) as error:
        print(f'data preparation failed: {error}', file=sys.stderr)
        return 1
    emit({'event': 'verified' if args.verify_only else 'ready',
          'manifest': str(args.output_dir.expanduser().resolve() / 'expert-manifest.json'),
          'trajectories': manifest['retained_count'], 'tasks': manifest['unique_tasks'],
          'downloaded_files': downloaded, 'source_revision': manifest['source_revision'],
          'summary': manifest['summary']})
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
