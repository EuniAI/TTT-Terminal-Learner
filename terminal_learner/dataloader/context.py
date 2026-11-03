"""Keep complete trajectory prefixes that fit the actual training serialization."""
from .trajectory import load_episode


def context_turn_limits(records, encoder, *, limit_turns=None, progress=None):
    limits, dropped = {}, []
    for index, record in enumerate(records, 1):
        episode = load_episode(record)
        turns = episode.turns[:limit_turns] if limit_turns is not None else episode.turns
        history = encoder.initial_history(episode.initial_observation)
        kept = 0
        for turn in turns:
            prompt, targets = encoder.action_example(episode.task, history, turn.action,
                                                     reasoning_content=turn.reasoning_content)
            kind, tokens = 'action', len(prompt) + len(targets)
            if tokens <= encoder.max_seq_len and turn.observation is not None:
                prompt, targets = encoder.observation_example(episode.task, history, turn.action,
                    turn.observation, reasoning_content=turn.reasoning_content)
                kind, tokens = 'observation', len(prompt) + len(targets)
            if tokens > encoder.max_seq_len:
                dropped.append(dict(trial=episode.trial, kept_turns=kept,
                    dropped_turns=len(turns) - kept, first_dropped_step_id=turn.step_id,
                    kind=kind, tokens=tokens))
                break
            kept += 1
            if turn.observation is not None:
                history = encoder.history(history, turn.action, turn.observation,
                                          reasoning_content=turn.reasoning_content)
        limits[episode.trial] = kept
        if progress is not None and (index % 25 == 0 or index == len(records)):
            progress(dict(event='context_scan', checked=index, total=len(records),
                          affected_episodes=len(dropped)))
    return limits, dropped


def apply_context_limits(records, limits):
    if set(limits) != {r['trial'] for r in records}:
        raise ValueError('saved context limits do not match selected trajectories')
    return [{**record, 'context_turn_limit': limits[record['trial']]} for record in records]
