import json
import logging
import queue
import threading
import time
from collections.abc import Callable

from .sensor import SensorEvent


def parse_sensor_message(
    line: str, *, clock: Callable[[], float] = time.monotonic
) -> SensorEvent:
    try:
        message = json.loads(line)
    except json.JSONDecodeError as error:
        raise ValueError("Sensor message must be valid JSON") from error

    if not isinstance(message, dict):
        raise ValueError("Sensor message must be a JSON object")

    sensor_id = message.get("sensor_id")
    active = message.get("active")
    sequence = message.get("sequence")
    if not isinstance(sensor_id, str) or not sensor_id.strip():
        raise ValueError("Sensor message must include a non-empty sensor_id")
    if not isinstance(active, bool):
        raise ValueError("Sensor message active field must be a boolean")
    if sequence is not None and (
        isinstance(sequence, bool) or not isinstance(sequence, int)
    ):
        raise ValueError("Sensor message sequence must be an integer")

    return SensorEvent(
        sensor_id=sensor_id.strip(),
        active=active,
        received_at=clock(),
        sequence=sequence,
    )


def _open_serial(**kwargs):
    import serial

    return serial.Serial(**kwargs)


class SerialSensorSource:
    def __init__(
        self,
        port: str,
        baudrate: int = 115200,
        *,
        read_timeout: float = 0.25,
        reconnect_initial_delay: float = 1.0,
        reconnect_max_delay: float = 15.0,
        serial_factory: Callable[..., object] | None = None,
        logger: logging.Logger | None = None,
    ) -> None:
        if not port.strip():
            raise ValueError("Serial port cannot be empty")
        if read_timeout <= 0:
            raise ValueError("Read timeout must be greater than zero")
        if reconnect_initial_delay <= 0:
            raise ValueError("Initial reconnect delay must be greater than zero")
        if reconnect_max_delay < reconnect_initial_delay:
            raise ValueError("Maximum reconnect delay must be at least the initial delay")

        self._port = port
        self._baudrate = baudrate
        self._read_timeout = read_timeout
        self._reconnect_initial_delay = reconnect_initial_delay
        self._reconnect_max_delay = reconnect_max_delay
        self._serial_factory = serial_factory or _open_serial
        self._logger = logger or logging.getLogger(__name__)
        self._events: queue.SimpleQueue[SensorEvent] = queue.SimpleQueue()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._read_loop,
            name=f"esp32-sensor-{self._port}",
            daemon=True,
        )
        self._thread.start()

    def drain_events(self) -> list[SensorEvent]:
        events = []
        while True:
            try:
                events.append(self._events.get_nowait())
            except queue.Empty:
                return events

    def close(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=self._read_timeout + 1.0)
            if self._thread.is_alive():
                self._logger.warning("ESP32 serial reader did not stop before timeout")

    def _read_loop(self) -> None:
        retry_delay = self._reconnect_initial_delay
        while not self._stop_event.is_set():
            connection = None
            try:
                connection = self._serial_factory(
                    port=self._port,
                    baudrate=self._baudrate,
                    timeout=self._read_timeout,
                )
                self._logger.info("Connected to ESP32 sensor on %s", self._port)
                while not self._stop_event.is_set():
                    raw_line = connection.readline()
                    if not raw_line:
                        continue
                    try:
                        line = (
                            raw_line.decode("utf-8").strip()
                            if isinstance(raw_line, bytes)
                            else str(raw_line).strip()
                        )
                        if line:
                            self._events.put(parse_sensor_message(line))
                    except (UnicodeDecodeError, ValueError):
                        self._logger.warning(
                            "Ignoring invalid ESP32 sensor message", exc_info=True
                        )
            except Exception:
                if not self._stop_event.is_set():
                    self._logger.exception(
                        "ESP32 sensor connection failed; retrying in %.1f seconds",
                        retry_delay,
                    )
            finally:
                if connection is not None:
                    try:
                        connection.close()
                    except Exception:
                        self._logger.warning(
                            "Failed to close ESP32 serial connection", exc_info=True
                        )

            if self._stop_event.wait(retry_delay):
                return
            retry_delay = min(retry_delay * 2, self._reconnect_max_delay)