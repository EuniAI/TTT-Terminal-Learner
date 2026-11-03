from __future__ import annotations

from dataclasses import dataclass

@dataclass
class Turn:
    action: str
    observation: str | None
    step_id: int
    reasoning_content: str | None = None


@dataclass
class Episode:
    task: str
    initial_observation: str
    turns: list[Turn]
    trial: str
    task_id: str
    reward: float | None
