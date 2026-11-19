import csv
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np

from screen_recording import ScreenRecorder, draw_recording_indicator


class FakeClock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now


class FakeVideoWriter:
    def __init__(self, path, fourcc, fps, frame_size):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.touch()
        self.writer_options = (fourcc, fps, frame_size)
        self.frames = []
        self.released = False

    def isOpened(self):
        return True

    def write(self, frame):
        self.frames.append(frame.copy())

    def release(self):
        self.released = True


class FailedVideoWriter(FakeVideoWriter):
    def isOpened(self):
        return False


class ScreenRecordingTests(unittest.TestCase):
    def test_video_frame_rate_stays_close_to_real_time(self):
        clock = FakeClock()
        writers = []

        def make_writer(*args):
            writer = FakeVideoWriter(*args)
            writers.append(writer)
            return writer

        frame = np.zeros((48, 64, 3), dtype=np.uint8)
        with TemporaryDirectory() as temporary_directory:
            with patch(
                "screen_recording.cv2.VideoWriter", side_effect=make_writer
            ):
                recorder = ScreenRecorder(
                    temporary_directory,
                    "fine-tuned",
                    output_fps=10,
                    clock=clock,
                )
                recorder.start((64, 48))
                for _ in range(5):
                    recorder.add_frame(frame)
                    clock.now += 0.5
                recorder.stop()

        self.assertEqual(len(writers[0].frames), 21)
        self.assertTrue(writers[0].released)

    def test_events_are_logged_with_video_time(self):
        clock = FakeClock()
        with TemporaryDirectory() as temporary_directory:
            with patch(
                "screen_recording.cv2.VideoWriter",
                side_effect=FakeVideoWriter,
            ):
                recorder = ScreenRecorder(
                    temporary_directory,
                    "yolo-world",
                    clock=clock,
                )
                recorder.start((64, 48))
                clock.now = 3.25
                recorder.log_event("points_awarded", "Ali", "total=10")
                recorder.stop()
            with recorder.events_path.open(encoding="utf-8") as event_file:
                rows = list(csv.reader(event_file))

        self.assertEqual(
            rows[0], ["video_time_sec", "event", "user", "detail"]
        )
        self.assertEqual(
            rows[1], ["3.25", "points_awarded", "Ali", "total=10"]
        )
        self.assertIn("yolo-world", recorder.video_path.name)

    def test_toggle_starts_and_stops_recording(self):
        writers = []

        def make_writer(*args):
            writer = FakeVideoWriter(*args)
            writers.append(writer)
            return writer

        with TemporaryDirectory() as temporary_directory:
            with patch("screen_recording.cv2.VideoWriter", side_effect=make_writer):
                recorder = ScreenRecorder(temporary_directory, "fine-tuned")
                self.assertTrue(recorder.toggle((64, 48)))
                self.assertTrue(recorder.is_recording)
                self.assertFalse(recorder.toggle((64, 48)))
                self.assertFalse(recorder.is_recording)
                self.assertTrue(recorder.video_path.exists())

        self.assertTrue(writers[0].released)

    def test_idle_recorder_writes_nothing(self):
        with TemporaryDirectory() as temporary_directory:
            recorder = ScreenRecorder(temporary_directory, "fine-tuned")
            frame = np.zeros((48, 64, 3), dtype=np.uint8)

            self.assertEqual(recorder.add_frame(frame), 0)
            recorder.log_event("drop_detected")
            self.assertIsNone(recorder.stop())
            self.assertEqual(list(Path(temporary_directory).iterdir()), [])

    def test_failed_writer_is_released_and_removed(self):
        writers = []

        def make_writer(*args):
            writer = FailedVideoWriter(*args)
            writers.append(writer)
            return writer

        with TemporaryDirectory() as temporary_directory:
            with patch("screen_recording.cv2.VideoWriter", side_effect=make_writer):
                recorder = ScreenRecorder(temporary_directory, "fine-tuned")
                with self.assertRaises(RuntimeError):
                    recorder.start((64, 48))

            self.assertFalse(recorder.is_recording)
            self.assertEqual(list(Path(temporary_directory).iterdir()), [])

        self.assertTrue(writers[0].released)

    def test_indicator_is_drawn_only_while_recording(self):
        frame = np.zeros((100, 200, 3), dtype=np.uint8)
        with TemporaryDirectory() as temporary_directory:
            with patch(
                "screen_recording.cv2.VideoWriter",
                side_effect=FakeVideoWriter,
            ):
                recorder = ScreenRecorder(temporary_directory, "fine-tuned")
                draw_recording_indicator(frame, recorder)
                self.assertFalse(frame.any())

                recorder.start((200, 100))
                draw_recording_indicator(frame, recorder)
                recorder.stop()

        self.assertTrue(frame.any())


if __name__ == "__main__":
    unittest.main()
