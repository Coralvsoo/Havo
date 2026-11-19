import csv
import json
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hardware.mock_sensor import MockSensorSource
from scripts import evaluate_session
from scripts.evaluate_session import calculate_metrics, replay_segment, truth_times
from scripts.review_session import read_review_events, read_labels, write_labels
from session_metrics import match_events


class MatchEventsTests(unittest.TestCase):
    def test_exact_matching(self):
        matches, fp, fn = match_events([1.0, 2.0], [1.0, 2.0], 0.1)
        self.assertEqual(matches, [(0, 0, 0.0), (1, 1, 0.0)])
        self.assertEqual(fp, [])
        self.assertEqual(fn, [])

    def test_outside_tolerance_is_fp_and_fn(self):
        matches, fp, fn = match_events([1.0], [2.5], 0.5)
        self.assertEqual(matches, [])
        self.assertEqual(fp, [0])
        self.assertEqual(fn, [0])

    def test_no_duplicate_matching_uses_smallest_difference(self):
        matches, fp, fn = match_events([1.0, 1.2], [1.1], 0.2)
        self.assertEqual(len(matches), 1)
        self.assertEqual(matches[0][:2], (1, 0))
        self.assertAlmostEqual(matches[0][2], 0.1)
        self.assertEqual(fp, [0])
        self.assertEqual(fn, [])


class MetricsTests(unittest.TestCase):
    def test_metrics_and_mean_detection_delay(self):
        metrics = calculate_metrics([1.2, 5.0], [1.0, 2.0], 0.5)
        self.assertEqual(metrics["TP"], 1)
        self.assertEqual(metrics["FP"], 1)
        self.assertEqual(metrics["FN"], 1)
        self.assertAlmostEqual(metrics["precision"], 0.5)
        self.assertAlmostEqual(metrics["recall"], 0.5)
        self.assertAlmostEqual(metrics["F1"], 0.5)
        self.assertAlmostEqual(metrics["mean_detection_delay"], 0.2)

    def test_mock_sensor_events_match_and_report_missed_drops(self):
        now = [5.0]
        source = MockSensorSource(clock=lambda: now[0])
        source.start()
        source.emit("bin", True)
        detected_event = source.drain_events()[0]
        matches, false_positives, false_negatives = match_events(
            [5.0], [detected_event.received_at], 0.1
        )
        self.assertEqual(len(matches), 1)
        self.assertEqual(false_positives, [])
        self.assertEqual(false_negatives, [])

        now[0] = 8.0
        source.emit("bin", True)
        missed_event = source.drain_events()[0]
        matches, false_positives, false_negatives = match_events(
            [], [missed_event.received_at], 0.1
        )
        self.assertEqual(matches, [])
        self.assertEqual(false_positives, [])
        self.assertEqual(false_negatives, [0])
        source.close()

    def test_sensor_ground_truth_takes_precedence_even_without_sensor_drops(self):
        with TemporaryDirectory() as temporary_directory:
            session_dir = Path(temporary_directory)
            (session_dir / "session.json").write_text(
                json.dumps({"sensor_connected": True}), encoding="utf-8"
            )
            (session_dir / "events.csv").write_text(
                "event_id,event_type,time_sec\n"
                "event_1,sensor_drop,1.0\n"
                "event_2,drop_detected,2.0\n",
                encoding="utf-8",
            )
            (session_dir / "labels.csv").write_text(
                "event_id,event_type,time_sec,label\n"
                "event_2,drop_detected,2.0,1\n",
                encoding="utf-8",
            )

            self.assertEqual(
                truth_times(session_dir), ([1.0], "sensor events")
            )

    def test_human_labels_and_review_order_use_saved_event_times(self):
        with TemporaryDirectory() as temporary_directory:
            session_dir = Path(temporary_directory)
            (session_dir / "session.json").write_text("{}", encoding="utf-8")
            with (session_dir / "events.csv").open(
                "w", newline="", encoding="utf-8"
            ) as event_file:
                writer = csv.DictWriter(
                    event_file, fieldnames=["event_id", "event_type", "time_sec"]
                )
                writer.writeheader()
                writer.writerows(
                    [
                        {
                            "event_id": "zone",
                            "event_type": "zone_activity",
                            "time_sec": "1.5",
                        },
                        {
                            "event_id": "drop",
                            "event_type": "drop_detected",
                            "time_sec": "2.5",
                        },
                    ]
                )
            (session_dir / "labels.csv").write_text(
                "event_id,event_type,time_sec,label\n"
                "zone,zone_activity,1.5,1\n"
                "drop,drop_detected,2.5,0\n",
                encoding="utf-8",
            )

            events = read_review_events(session_dir)
            self.assertEqual([event["event_id"] for event in events], ["drop", "zone"])
            self.assertEqual(
                truth_times(session_dir), ([1.5], "human labels")
            )
            labels_path = session_dir / "labels.csv"
            labels = read_labels(labels_path)
            write_labels(labels_path, events, labels)
            self.assertEqual(read_labels(labels_path), labels)


class ReplayTests(unittest.TestCase):
    class FakeCapture:
        def __init__(self, _path):
            self.frames = [np.zeros((24, 32, 3), dtype=np.uint8) for _ in range(5)]

        def isOpened(self):
            return True

        def get(self, _property):
            return 30.0

        def read(self):
            if not self.frames:
                return False, None
            return True, self.frames.pop(0)

        def release(self):
            pass

    @staticmethod
    def new_state():
        return {
            "detected_times": [],
            "tracks": {},
            "locked_box": None,
            "calibration_boxes": [],
            "calibration_count": 0,
            "drift": {"bad_checks": 0, "drifted": False},
        }

    def test_lock_replay_rechecks_bin_after_calibration(self):
        state = self.new_state()
        bin_detection = {
            "box": (2, 2, 20, 20),
            "confidence": 0.9,
            "kind": "bin",
            "label": "bin",
        }
        with TemporaryDirectory() as temporary_directory:
            segment = Path(temporary_directory) / "segment_000.mp4"
            segment.touch()
            with patch.object(
                evaluate_session.cv2, "VideoCapture", self.FakeCapture
            ), patch.object(
                evaluate_session.face, "CALIBRATION_FRAMES", 3
            ), patch.object(
                evaluate_session.face, "CALIBRATION_MIN_HITS", 2
            ), patch.object(
                evaluate_session.face, "detect_bins", return_value=[bin_detection]
            ) as detect_bins, patch.object(
                evaluate_session.face, "detect_items", return_value=[]
            ) as detect_items:
                replay_segment(
                    segment,
                    object(),
                    object(),
                    "lock",
                    segment_index=0,
                    frame_timestamps={},
                    time_base=0.0,
                    state=state,
                    fixed_bin_box=bin_detection["box"],
                )

        self.assertEqual(detect_bins.call_count, 5)
        self.assertEqual(detect_items.call_count, 3)

    def test_detect_and_fixed_replay_use_expected_detector_cadence(self):
        bin_detection = {
            "box": (2, 2, 20, 20),
            "confidence": 0.9,
            "kind": "bin",
            "label": "bin",
        }
        for mode, expected_bin_calls in (("detect", 5), ("fixed", 0)):
            with self.subTest(mode=mode), TemporaryDirectory() as temporary_directory:
                segment = Path(temporary_directory) / "segment_000.mp4"
                segment.touch()
                with patch.object(
                    evaluate_session.cv2, "VideoCapture", self.FakeCapture
                ), patch.object(
                    evaluate_session.face,
                    "detect_bins",
                    return_value=[bin_detection],
                ) as detect_bins, patch.object(
                    evaluate_session.face, "detect_items", return_value=[]
                ) as detect_items:
                    replay_segment(
                        segment,
                        object(),
                        object(),
                        mode,
                        segment_index=0,
                        frame_timestamps={},
                        time_base=0.0,
                        state=self.new_state(),
                        fixed_zone=(2, 2, 20, 20),
                        fixed_bin_box=bin_detection["box"],
                    )

                self.assertEqual(detect_bins.call_count, expected_bin_calls)
                self.assertEqual(detect_items.call_count, 5)


if __name__ == "__main__":
    unittest.main()
