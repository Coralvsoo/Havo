import logging
import queue
import threading
import time
import unittest
from unittest.mock import Mock

from hardware import MockSensorSource, SerialSensorSource
from hardware.esp32_serial import parse_sensor_message


class FakeSerial:
    def __init__(self, lines):
        self._lines = queue.Queue()
        for line in lines:
            self._lines.put(line)
        self.closed = False

    def readline(self):
        try:
            return self._lines.get(timeout=0.02)
        except queue.Empty:
            return b""

    def close(self):
        self.closed = True


class SensorModuleTests(unittest.TestCase):
    def test_parse_sensor_message_validates_fields_and_sets_timestamp(self):
        event = parse_sensor_message(
            '{"sensor_id":"bin_entry","active":true,"sequence":7}',
            clock=lambda: 12.5,
        )

        self.assertEqual(event.sensor_id, "bin_entry")
        self.assertTrue(event.active)
        self.assertEqual(event.sequence, 7)
        self.assertEqual(event.received_at, 12.5)
        with self.assertRaises(ValueError):
            parse_sensor_message('{"sensor_id":"bin_entry","active":"yes"}')

    def test_mock_source_emits_and_drains_events(self):
        source = MockSensorSource(clock=lambda: 3.0)
        source.start()
        source.emit("bin_entry", True, sequence=2)

        self.assertEqual(
            source.drain_events(),
            [
                parse_sensor_message(
                    '{"sensor_id":"bin_entry","active":true,"sequence":2}',
                    clock=lambda: 3.0,
                )
            ],
        )
        self.assertEqual(source.drain_events(), [])
        source.close()

    def test_serial_source_retries_connection_failures_and_delivers_events(self):
        attempts = 0
        serial_connection = FakeSerial(
            [b'{"sensor_id":"bin_entry","active":true,"sequence":1}\n']
        )
        logger = Mock(spec=logging.Logger)

        def serial_factory(**_kwargs):
            nonlocal attempts
            attempts += 1
            if attempts == 1:
                raise OSError("ESP32 is not connected")
            return serial_connection

        source = SerialSensorSource(
            "COM_TEST",
            read_timeout=0.02,
            reconnect_initial_delay=0.01,
            reconnect_max_delay=0.02,
            serial_factory=serial_factory,
            logger=logger,
        )
        source.start()
        received = []
        deadline = time.monotonic() + 1.0
        while not received and time.monotonic() < deadline:
            received.extend(source.drain_events())
            threading.Event().wait(0.01)
        source.close()

        self.assertEqual(len(received), 1)
        self.assertEqual(received[0].sensor_id, "bin_entry")
        self.assertGreaterEqual(attempts, 2)
        logger.exception.assert_called_once()
        self.assertTrue(serial_connection.closed)


if __name__ == "__main__":
    unittest.main()