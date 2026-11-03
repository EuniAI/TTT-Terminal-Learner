from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path
from terminal_learner.model.data import Turn, Episode
from terminal_learner.protocol import canonical_action

def load_episode(record):
    data = json.loads(Path(record['trajectory_file']).read_text())
    if data['agent']['name'] != 'terminus-2' or data['agent']['model_name'] != 'anthropic/claude-opus-4-6':
        raise ValueError(f'non-expert source: {record["trial"]}')
    steps = data['steps']
    first = next((s['message'] for s in steps if s.get('source') == 'user'), None)
    if not isinstance(first, str):
        raise ValueError('missing task text')
    task = first.split('Task Description:\n', 1)[-1]
    parts = task.split('\nCurrent terminal state:', 1)
    task, initial = parts[0].strip(), parts[1].strip() if len(parts) == 2 else ''
    turns = []
    for step in steps:
        if step.get('source') != 'agent':
            continue
        calls = step.get('tool_calls', [])
        if not calls:
            # Parser failures/reasoning-only records are not fabricated shell actions.
            continue
        commands, complete = [], False
        for call in calls:
            if call['function_name'] == 'bash_command':
                args = call['arguments']
                commands.append({'keystrokes': args['keystrokes'],
                                 'duration': args.get('duration', 1.0)})
            elif call['function_name'] == 'mark_task_complete':
                complete = True
            else:
                raise ValueError(f'unknown action tool: {call["function_name"]}')
        results = (step.get('observation') or {}).get('results', [])
        if any(not isinstance(x.get('content'), str) for x in results):
            raise ValueError('non-text observation requires a dedicated adapter')
        observation = '\n'.join(x['content'] for x in results) if results else None
        reasoning = step.get('reasoning_content')
        if reasoning is not None and not isinstance(reasoning, str):
            raise ValueError('reasoning_content must be text or null')
        turns.append(Turn(canonical_action({'commands': commands, 'task_complete': complete}),
                          observation, step['step_id'], reasoning))
    if 'context_turn_limit' in record:
        limit = record['context_turn_limit']
        if type(limit) is not int or not 0 <= limit <= len(turns):
            raise ValueError('invalid saved context turn limit')
        turns = turns[:limit]
    return Episode(task, initial, turns, record['trial'], record['task'], record.get('reward'))


def read_manifest(path):
    """Resolve stored paths independently of the caller's current directory."""
    path = Path(path).resolve()
    manifest = json.loads(path.read_text())
    records = manifest['records']
    trials = [r['trial'] for r in records]
    if len(set(trials)) != len(trials):
        raise ValueError('duplicate trial IDs in manifest')
    result = []
    for row in records:
        r = dict(row)
        for key in ('trajectory_file', 'result_file'):
            value = Path(r[key])
            candidates = [value] if value.is_absolute() else [parent / value for parent in path.parents]
            found = next((p for p in candidates if p.is_file()), None)
            if found is None:
                raise FileNotFoundError(value)
            digest = hashlib.sha256(found.read_bytes()).hexdigest()
            expected = r.get(key.replace('_file', '_sha256'))
            if expected is not None and digest != expected:
                raise ValueError(f'data hash mismatch: {found}')
            r[key], r[key.replace('_file', '_sha256')] = str(found), digest
        result.append(r)
    fingerprint = hashlib.sha256(json.dumps([
        (r['trial'], r['task'], r['trajectory_sha256'], r['result_sha256'])
        for r in sorted(result, key=lambda x: x['trial'])]).encode()).hexdigest()
    return result, fingerprint

def split_tasks(records, *, test_fraction=0.0, seed=0):
    """Group all attempts of one task, including different reward outcomes."""
    if not 0 <= test_fraction < 1:
        raise ValueError('test_fraction must be in [0, 1)')
    tasks = sorted({r['task'] for r in records})
    if not tasks:
        raise ValueError('empty dataset')
    if test_fraction == 0:
        return tasks, []
    if len(tasks) < 2:
        raise ValueError('at least two distinct tasks are needed for held-out evaluation')
    random.Random(seed).shuffle(tasks)
    n_test = max(1, min(len(tasks) - 1, round(len(tasks) * test_fraction)))
    return sorted(tasks[n_test:]), sorted(tasks[:n_test])
