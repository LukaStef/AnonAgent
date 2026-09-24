"""Structured record of what the agent did, step by step.

The console demo and the Streamlit UI both render this list rather than
printing as they go, so the two views can never drift apart.

Events carry a ``sensitive`` flag. Anything marked sensitive holds real
private values and belongs only in the local panel -- never in the view that
shows what was sent to the cloud.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Iterator


@dataclass(frozen=True)
class AgentEvent:
    step: str
    message: str
    timestamp: datetime
    payload: dict[str, Any] = field(default_factory=dict)
    sensitive: bool = False


class EventLog:
    """An append-only list of agent events."""

    def __init__(self) -> None:
        self._events: list[AgentEvent] = []

    def record(
        self,
        step: str,
        message: str,
        payload: dict[str, Any] | None = None,
        *,
        sensitive: bool = False,
    ) -> AgentEvent:
        event = AgentEvent(
            step=step,
            message=message,
            timestamp=datetime.now(timezone.utc),
            payload=payload or {},
            sensitive=sensitive,
        )
        self._events.append(event)
        return event

    @property
    def events(self) -> list[AgentEvent]:
        return list(self._events)

    def public_events(self) -> list[AgentEvent]:
        """Only the events safe to show outside the local trust boundary."""
        return [event for event in self._events if not event.sensitive]

    def __iter__(self) -> Iterator[AgentEvent]:
        return iter(self._events)

    def __len__(self) -> int:
        return len(self._events)
