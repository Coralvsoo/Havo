import csv
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np

from session_recording import SessionRecorder, blur_faces


class FakeVideoWriter:
    def __init__(self, path, _fourcc, _fps, _frame_size):
        self.path = Path(path)
        self.path.write_bytes(b"")

    def isOpened(self):
        return True

    def write(self, _frame):
        with self.path.open("ab") as output:
            output.write(b"frame")

    def release(self):
        pass


class FakeCascade:
    def empty(self):
        return False

    def detectMultiScale(self, _gray, **_kwargs):
        return [(5, 5, 30, 30)]


class SessionRecordingTests(unittest.TestCase):
    def test_blur_faces_changes_pixels_only_inside_face_box(self):
        frame = np.zeros((50, 50, 3), dtype=np.uint8)
        frame[5:35, 5:35] = np.arange(30, dtype=np.uint8)[:, None, None]

        blurred = blur_faces(frame, FakeCascade())

        self.assertFalse(np.array_equal(blurred[5:35, 5:35], frame[5:35, 5:35]))
        np.testing.assert_array_equal(blurred[:5], frame[:5])
        np.testing.assert_array_equal(blurred[35:], frame[35:])
        np.testing.assert_array_equal(blurred[:, :5], frame[:, :5])
        np.testing.assert_array_equal(blurred[:, 35:], frame[:, 35:])

    def test_recording_creates_session_files_and_event_clip(self):
        now = [100.0]
        with TemporaryDirectory() as temporary_directory:
            recorder = SessionRecorder(
                temporary_directory,
                (32, 24),
                2,
                {"zone_mode": "lock", "calibrated_bin_box": [1, 2, 20, 20]},
                max_storage_bytes=100000,
                blur=False,
                clip_seconds_before=1,
                clip_seconds_after=1,
                segment_minutes=5,
                clock=lambda: now[0],
                video_writer_factory=FakeVideoWriter,
            )
            for value in range(5):
                now[0] += 0.5
                frame = np.full((24, 32, 3), value, dtype=np.uint8)
                frame_info = recorder.add_frame(frame)
                if value == 2:
                    recorder.record_event(
                        "drop_detected",
                        time_sec=frame_info["time_sec"],
                        global_frame=frame_info["global_frame"],
                        zone=(2, 3, 10, 12),
                        zone_mode="lock",
                        user="Alice",
                        item_labels=["bottle"],
                    )
            recorder.close()

            session_directories = list(Path(temporary_directory).iterdir())
            self.assertEqual(len(session_directories), 1)
            session_directory = session_directories[0]
            self.assertTrue(
                (session_directory / "segments" / "segment_000.mp4").is_file()
            )
            self.assertTrue(
                (session_directory / "segments" / "frames.csv").is_file()
            )
            self.assertTrue((session_directory / "events.csv").is_file())
            self.assertTrue((session_directory / "session.json").is_file())
            clip_path = session_directory / "clips" / "event_000001.mp4"
            self.assertTrue(clip_path.is_file())
            self.assertEqual(clip_path.stat().st_size, 5 * len(b"frame"))
            clip_metadata = json.loads(
                (session_directory / "clips" / "event_000001.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(clip_metadata["event_type"], "drop_detected")
            self.assertEqual(clip_metadata["item_labels"], ["bottle"])
            with (session_directory / "events.csv").open(
                newline="", encoding="utf-8"
            ) as event_file:
                event_rows = list(csv.DictReader(event_file))
            self.assertEqual(event_rows[0]["user"], "Alice")

    def test_storage_limit_deletes_oldest_segment_but_keeps_clip(self):
        now = [0.0]
        with TemporaryDirectory() as temporary_directory:
            recorder = SessionRecorder(
                temporary_directory,
                (8, 8),
                1,
                {"zone_mode": "fixed"},
                max_storage_bytes=100000,
                blur=False,
                clip_seconds_before=0,
                clip_seconds_after=0,
                segment_minutes=1 / 60,
                clock=lambda: now[0],
                video_writer_factory=FakeVideoWriter,
            )
            recorder.add_frame(np.zeros((8, 8, 3), dtype=np.uint8))
            recorder.record_event(
                "drop_detected", time_sec=recorder.time_sec, global_frame=0
            )
            now[0] += 1
            recorder.add_frame(np.zeros((8, 8, 3), dtype=np.uint8))
            first_segment = recorder.segment_directory / "segment_000.mp4"
            baseline_size = sum(
                path.stat().st_size
                for path in recorder.directory.rglob("*")
                if path.is_file()
            ) - first_segment.stat().st_size
            recorder.max_storage_bytes = (
                baseline_size + first_segment.stat().st_size - 1
            )
            recorder.close()

            segments = list(recorder.segment_directory.glob("segment_*.mp4"))
            self.assertEqual([path.name for path in segments], ["segment_001.mp4"])
            self.assertTrue(
                (recorder.clip_directory / "event_000001.mp4").is_file()
            )
            self.assertTrue((recorder.directory / "events.csv").is_file())

    def test_blurring_is_enabled_by_default_and_requires_valid_cascade(self):
        with TemporaryDirectory() as temporary_directory:
            with patch("session_recording.cv2.CascadeClassifier") as classifier:
                classifier.return_value.empty.return_value = True
                with self.assertRaisesRegex(RuntimeError, "face-blur cascade"):
                    SessionRecorder(
                        temporary_directory,
                        (8, 8),
                        1,
                        {},
                        max_storage_bytes=100,
                    )


if __name__ == "__main__":
    unittest.main()
