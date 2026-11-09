from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class SensorEvent:
    sensor_id: str
    active: bool
    received_at: float
    sequence: int | None = None


class SensorSource(Protocol):
    def start(self) -> None:
        """Start receiving sensor events."""

    def drain_events(self) -> list[SensorEvent]:
        """Return all currently available events without blocking."""

    def close(self) -> None:
        """Stop receiving events and release resources."""