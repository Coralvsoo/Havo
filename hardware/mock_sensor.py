import queue
import time
from collections.abc import Callable

from .sensor import SensorEvent


class MockSensorSource:
    def __init__(self, *, clock: Callable[[], float] = time.monotonic) -> None:
        self._clock = clock
        self._events: queue.SimpleQueue[SensorEvent] = queue.SimpleQueue()
        self._started = False

    def start(self) -> None:
        self._started = True

    def emit(
        self, sensor_id: str, active: bool, sequence: int | None = None
    ) -> None:
        if not self._started:
            raise RuntimeError("Mock sensor source must be started before emitting")
        self._events.put(
            SensorEvent(
                sensor_id=sensor_id,
                active=active,
                received_at=self._clock(),
                sequence=sequence,
            )
        )

    def drain_events(self) -> list[SensorEvent]:
        events = []
        while True:
            try:
                events.append(self._events.get_nowait())
            except queue.Empty:
                return events

    def close(self) -> None:
        self._started = False