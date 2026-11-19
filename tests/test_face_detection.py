import csv
import json
import time
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import MagicMock, patch

import numpy as np

import face
from hardware.sensor import SensorEvent


class FakeBox:
    def __init__(self, class_id, confidence, coordinates):
        self.cls = [np.array(class_id)]
        self.conf = [np.array(confidence)]
        self.xyxy = [np.array(coordinates, dtype=np.float32)]


class FakeResult:
    names = {
        0: "bottle",
        1: "trash can",
        2: "person",
        3: "recycling bin",
    }

    def __init__(self, boxes):
        self.boxes = boxes


class FakeModel:
    def __init__(self, result):
        self.result = result

    @property
    def names(self):
        return self.result.names

    def set_classes(self, classes):
        self.classes = classes

    def __call__(self, _frame, verbose=False, conf=None):
        return [self.result]


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


class SingleEventSensor:
    def __init__(self):
        self.drained = False

    def start(self):
        pass

    def drain_events(self):
        if self.drained:
            return []
        self.drained = True
        return [SensorEvent("bin", True, time.monotonic())]

    def close(self):
        pass


class FaceDetectionTests(unittest.TestCase):
    def test_box_iou_and_median_box(self):
        self.assertAlmostEqual(
            face.box_iou((0, 0, 10, 10), (5, 0, 15, 10)), 1 / 3
        )
        self.assertEqual(
            face.median_box([(0, 0, 10, 10), (2, 2, 12, 12), (4, 4, 14, 14)]),
            (2, 2, 12, 12),
        )
        self.assertEqual(face.box_iou((0, 0, 0, 0), (0, 0, 0, 0)), 0.0)

    def test_match_bin_detection_follows_the_selected_bin(self):
        target = {
            "box": (12, 10, 112, 110),
            "confidence": 0.6,
        }
        unrelated = {
            "box": (300, 200, 400, 300),
            "confidence": 0.99,
        }

        self.assertIs(
            face.match_bin_detection([unrelated, target], (10, 10, 110, 110)),
            target,
        )
        self.assertIsNone(
            face.match_bin_detection([unrelated], (10, 10, 110, 110))
        )

    def test_run_session_requires_a_registered_face(self):
        with patch.object(face, "load_embeddings", return_value=[]):
            with self.assertRaisesRegex(RuntimeError, "Face recognition is required"):
                face.run_session(face.MODEL_OLD)

    def test_decide_calibration_requires_enough_stable_boxes(self):
        stable_boxes = [(10 + index % 2, 10, 110 + index % 2, 110) for index in range(24)]
        calibrated_box, reason = face.decide_calibration(stable_boxes)

        self.assertIsNotNone(calibrated_box)
        self.assertEqual(reason, "ok")
        self.assertEqual(
            face.decide_calibration(stable_boxes[: face.CALIBRATION_MIN_HITS - 1]),
            (None, "too few detections"),
        )
        unstable_boxes = [
            (index * 100, 0, index * 100 + 50, 50)
            for index in range(face.CALIBRATION_MIN_HITS)
        ]
        self.assertEqual(
            face.decide_calibration(unstable_boxes), (None, "unstable box")
        )

    def test_update_drift_requires_three_bad_checks_and_is_sticky(self):
        state = {"bad_checks": 0, "drifted": False}
        state = face.update_drift(state, 0.8)
        self.assertEqual(state, {"bad_checks": 0, "drifted": False})
        state = face.update_drift(state, None)
        state = face.update_drift(state, 0.2)
        self.assertEqual(state, {"bad_checks": 2, "drifted": False})
        state = face.update_drift(state, 0.2)
        self.assertEqual(state, {"bad_checks": 3, "drifted": True})
        self.assertEqual(
            face.update_drift(state, 0.9), {"bad_checks": 3, "drifted": True}
        )

    def test_calibration_detects_only_during_calibration_frames(self):
        frame = np.zeros((40, 60, 3), dtype=np.uint8)
        camera = MagicMock()
        camera.read.return_value = (True, frame)
        bin_model = MagicMock(
            return_value=[FakeResult([FakeBox(0, 0.9, (5, 5, 30, 30))])]
        )
        bin_model.return_value[0].names = {0: "bin"}

        with patch.object(face, "CALIBRATION_FRAMES", 3):
            with patch.object(face, "CALIBRATION_MIN_HITS", 2):
                box, reason, _, _ = face.calibrate_bin(
                    camera, bin_model, frame_count=3
                )

        self.assertEqual(box, (5, 5, 30, 30))
        self.assertEqual(reason, "ok")
        self.assertEqual(bin_model.call_count, 3)

    def test_lock_mode_tracks_bin_after_calibration(self):
        frame = np.zeros((40, 60, 3), dtype=np.uint8)
        camera = MagicMock()
        camera.isOpened.return_value = True
        camera.get.return_value = 30.0
        camera.read.side_effect = [(True, frame.copy()) for _ in range(3)]
        item_model = MagicMock()

        def bin_result(box):
            result = FakeResult([FakeBox(0, 0.9, box)])
            result.names = {0: "bin"}
            return [result]

        bin_model = MagicMock(
            side_effect=[
                bin_result((5, 5, 30, 30)),
                bin_result((5, 5, 30, 30)),
                bin_result((5, 5, 30, 30)),
                bin_result((7, 5, 32, 30)),
            ]
        )

        with patch.object(
            face, "CALIBRATION_FRAMES", 3
        ), patch.object(
            face, "CALIBRATION_MIN_HITS", 2
        ), patch.object(
            face, "load_litter_models", return_value=(item_model, bin_model)
        ), patch.object(
            face, "load_embeddings", return_value=[("Alice", np.array([1.0]))]
        ), patch.object(
            face.cv2, "VideoCapture", return_value=camera
        ), patch.object(
            face,
            "confirm_bin",
            return_value={
                "box": (5, 5, 30, 30),
                "label": "calibrated bin",
                "confidence": 1.0,
                "kind": "bin",
                "center": (17, 17),
                "inside": False,
            },
        ), patch.object(
            face, "detect_items", return_value=[]
        ) as detect_items, patch.object(
            face.cv2, "imshow"
        ), patch.object(
            face.cv2, "waitKey", return_value=ord("q")
        ), patch.object(
            face.cv2, "destroyAllWindows"
        ):
            face.run_session(face.MODEL_FINE_TUNED)

        self.assertEqual(bin_model.call_count, 4)
        self.assertEqual(
            detect_items.call_args.args[2], (8, 0, 30, 15)
        )

    def test_recorded_session_saves_confirmed_drop_event(self):
        frame = np.zeros((40, 60, 3), dtype=np.uint8)
        camera = MagicMock()
        camera.isOpened.return_value = True
        camera.get.return_value = 30.0
        camera.read.side_effect = [(True, frame.copy()) for _ in range(5)]
        target_bin = {
            "box": (5, 5, 55, 35),
            "label": "selected bin",
            "confidence": 1.0,
            "kind": "bin",
            "center": (30, 20),
            "inside": False,
        }
        centers = [(25, -2), (25, 8), (25, 9), (25, 9), (25, 9)]
        item_frames = [
            [
                {
                    "box": (center[0] - 2, center[1] - 2, center[0] + 2, center[1] + 2),
                    "label": "bottle",
                    "confidence": 0.9,
                    "kind": "recyclable",
                    "center": center,
                    "inside": False,
                }
            ]
            for center in centers
        ]
        session_recorder = face.SessionRecorder

        def make_recorder(*args, **kwargs):
            return session_recorder(
                *args, video_writer_factory=FakeVideoWriter, **kwargs
            )

        with TemporaryDirectory() as temporary_directory:
            with patch.object(
                face, "load_litter_models", return_value=(MagicMock(), None)
            ), patch.object(
                face, "load_embeddings", return_value=[("Alice", np.array([1.0]))]
            ), patch.object(
                face.cv2, "VideoCapture", return_value=camera
            ), patch.object(
                face, "select_target_bin", return_value=target_bin
            ), patch.object(
                face, "detect_items", side_effect=item_frames
            ), patch.object(
                face, "SessionRecorder", side_effect=make_recorder
            ), patch.object(
                face.cv2, "imshow"
            ), patch.object(
                face.cv2, "waitKey", side_effect=[0, 0, 0, 0, ord("q")]
            ), patch.object(
                face.cv2, "destroyAllWindows"
            ):
                session_directory = face.run_session(
                    face.MODEL_OLD,
                    zone_mode=face.ZONE_FIXED,
                    record_directory=temporary_directory,
                    blur=False,
                )

            with (session_directory / "events.csv").open(
                newline="", encoding="utf-8"
            ) as event_file:
                event_rows = list(csv.DictReader(event_file))

            self.assertEqual(
                [row["event_type"] for row in event_rows],
                ["calibrated", "drop_detected"],
            )
            clip_path = session_directory / "clips" / "event_000002.mp4"
            self.assertTrue(clip_path.is_file())
            session_metadata = json.loads(
                (session_directory / "session.json").read_text(encoding="utf-8")
            )
            self.assertEqual(session_metadata["calibrated_bin_box"], [5, 5, 55, 35])

    def test_sensor_self_check_writes_live_report(self):
        frame = np.zeros((40, 60, 3), dtype=np.uint8)
        camera = MagicMock()
        camera.isOpened.return_value = True
        camera.get.return_value = 30.0
        camera.read.return_value = (True, frame)
        target_bin = {
            "box": (5, 5, 55, 35),
            "label": "selected bin",
            "confidence": 1.0,
            "kind": "bin",
            "center": (30, 20),
            "inside": False,
        }
        session_recorder = face.SessionRecorder

        def make_recorder(*args, **kwargs):
            return session_recorder(
                *args, video_writer_factory=FakeVideoWriter, **kwargs
            )

        with TemporaryDirectory() as temporary_directory:
            with patch.object(
                face, "load_litter_models", return_value=(MagicMock(), None)
            ), patch.object(
                face, "load_embeddings", return_value=[("Alice", np.array([1.0]))]
            ), patch.object(
                face.cv2, "VideoCapture", return_value=camera
            ), patch.object(
                face, "select_target_bin", return_value=target_bin
            ), patch.object(
                face, "detect_items", return_value=[]
            ), patch.object(
                face, "SessionRecorder", side_effect=make_recorder
            ), patch.object(
                face.cv2, "imshow"
            ), patch.object(
                face.cv2, "waitKey", return_value=ord("q")
            ), patch.object(
                face.cv2, "destroyAllWindows"
            ):
                session_directory = face.run_session(
                    face.MODEL_OLD,
                    zone_mode=face.ZONE_FIXED,
                    record_directory=temporary_directory,
                    blur=False,
                    sensor=SingleEventSensor(),
                )

            with (session_directory / "live_report.csv").open(
                newline="", encoding="utf-8"
            ) as report_file:
                report = list(csv.DictReader(report_file))

            self.assertEqual(report[0]["true_positive"], "0")
            self.assertEqual(report[0]["false_positive"], "0")
            self.assertEqual(report[0]["false_negative"], "1")

    def test_project_checkpoint_is_preferred_over_downloads_copy(self):
        project_checkpoint = face.PROJECT_DIR / "P04_bin_dataset2_best.pt"
        downloads_checkpoint = Path.home() / "Downloads" / "P04_bin_dataset2_best.pt"

        self.assertLess(
            face.FINE_TUNED_BIN_MODEL_PATHS.index(project_checkpoint),
            face.FINE_TUNED_BIN_MODEL_PATHS.index(downloads_checkpoint),
        )

    def test_load_old_model_sets_existing_yolo_world_classes(self):
        model = FakeModel(FakeResult([]))

        with patch.object(face, "YOLO", return_value=model) as yolo:
            item_model, bin_model = face.load_litter_models(face.MODEL_OLD)

        self.assertIs(item_model, model)
        self.assertIsNone(bin_model)
        yolo.assert_called_once_with("yolov8s-world.pt")
        self.assertEqual(list(model.classes), list(face.YOLO_CLASSES))

    def test_select_target_bin_shows_live_preview_and_returns_dragged_region(self):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        camera = MagicMock()
        camera.read.return_value = (True, frame)
        callbacks = []

        def capture_mouse_callback(_window_name, callback):
            callbacks.append(callback)

        def draw_selection_then_enter(_delay):
            callback = callbacks[0]
            callback(face.cv2.EVENT_LBUTTONDOWN, 100, 120, 0, None)
            callback(face.cv2.EVENT_MOUSEMOVE, 300, 360, 0, None)
            callback(face.cv2.EVENT_LBUTTONUP, 300, 360, 0, None)
            return 13

        with patch.object(face.cv2, "namedWindow") as named_window:
            with patch.object(face.cv2, "setMouseCallback", side_effect=capture_mouse_callback):
                with patch.object(face.cv2, "imshow") as show_frame:
                    with patch.object(
                        face.cv2, "waitKey", side_effect=draw_selection_then_enter
                    ):
                        with patch.object(face.cv2, "destroyWindow") as destroy_window:
                            target_bin = face.select_target_bin(camera)

        named_window.assert_called_once()
        show_frame.assert_called_once()
        destroy_window.assert_called_once()
        camera.read.assert_called_once()
        self.assertEqual(target_bin["box"], (100, 120, 300, 360))
        self.assertEqual(target_bin["center"], (200, 240))
        self.assertEqual(target_bin["label"], "selected bin")
        self.assertEqual(target_bin["kind"], "bin")

    def test_select_target_bin_can_be_cancelled(self):
        camera = MagicMock()
        camera.read.return_value = (
            True,
            np.zeros((480, 640, 3), dtype=np.uint8),
        )
        with patch.object(face.cv2, "namedWindow"):
            with patch.object(
                face.cv2,
                "setMouseCallback",
                return_value=None,
            ):
                with patch.object(face.cv2, "imshow"):
                    with patch.object(face.cv2, "waitKey", return_value=27):
                        with patch.object(face.cv2, "destroyWindow"):
                            with self.assertRaisesRegex(
                                RuntimeError, "selection was cancelled"
                            ):
                                face.select_target_bin(camera)

    def test_keep_selected_bin_ignores_model_bin_boxes(self):
        target_bin = {
            "box": (100, 120, 300, 360),
            "label": "selected bin",
            "confidence": 1.0,
            "kind": "bin",
            "center": (200, 240),
            "inside": False,
        }
        recyclable = {
            "box": (180, 90, 220, 130),
            "label": "bottle",
            "confidence": 0.9,
            "kind": "recyclable",
            "center": (200, 110),
            "inside": False,
        }
        false_bin_boxes = [
            {"kind": "bin", "label": "bin", "box": (0, 0, 640, 480)},
            {"kind": "bin", "label": "recycling bin", "box": (20, 30, 80, 90)},
        ]

        detections = face.keep_selected_bin(
            [recyclable, *false_bin_boxes], target_bin
        )

        self.assertEqual(detections, [recyclable, target_bin])
        self.assertIsNot(detections[-1], target_bin)

    def test_pretrained_bin_selection_prefers_recycling_bin(self):
        trash_bin = {"kind": "bin", "label": "trash can", "confidence": 0.9}
        recycling_bin = {
            "kind": "bin",
            "label": "recycling bin",
            "confidence": 0.6,
        }
        unrelated = {"kind": "recyclable", "label": "bottle"}

        self.assertIs(
            face.get_detected_bin([trash_bin, unrelated, recycling_bin]),
            recycling_bin,
        )
        self.assertIs(face.get_detected_bin([trash_bin, unrelated]), trash_bin)
        self.assertIsNone(face.get_detected_bin([unrelated]))

    def test_comparison_recognizes_face_on_both_detector_panels(self):
        frame = np.zeros((40, 60, 3), dtype=np.uint8)
        camera = MagicMock()
        camera.isOpened.return_value = True
        camera.read.side_effect = [(True, frame.copy()) for _ in range(11)]
        old_item_model = MagicMock()
        new_item_model = MagicMock()
        new_bin_model = MagicMock()
        embeddings = [("Alice", np.array([1.0], dtype=np.float32))]
        target_bin = {
            "box": (5, 5, 30, 35),
            "label": "selected bin",
            "confidence": 1.0,
            "kind": "bin",
            "center": (17, 20),
            "inside": False,
        }
        model_detections = [
            {
                "box": (0, 0, 60, 40),
                "label": "bin",
                "confidence": 0.99,
                "kind": "bin",
                "center": (30, 20),
                "inside": False,
            },
            {
                "box": (20, 10, 25, 20),
                "label": "bottle",
                "confidence": 0.9,
                "kind": "recyclable",
                "center": (22, 15),
                "inside": False,
            },
        ]

        with TemporaryDirectory() as temporary_directory:
            checkpoint = Path(temporary_directory) / "best.pt"
            checkpoint.write_bytes(b"test checkpoint")
            with patch.object(face, "FINE_TUNED_BIN_MODEL_PATH", checkpoint):
                with patch.object(face, "load_embeddings", return_value=embeddings):
                    with patch.object(
                        face,
                        "load_litter_models",
                        side_effect=[
                            (old_item_model, None),
                            (new_item_model, new_bin_model),
                        ],
                    ):
                        with patch.object(face.cv2, "VideoCapture", return_value=camera):
                            with patch.object(
                                face, "select_target_bin", return_value=target_bin
                            ) as select_target_bin:
                                with patch.object(
                                    face, "detect_litter", return_value=model_detections
                                ):
                                    with patch.object(face, "draw_litter") as draw_litter:
                                        with patch.object(
                                            face,
                                            "recognize_frame",
                                            return_value=(
                                                "Identified: Alice",
                                                0.1,
                                                {"x": 1, "y": 2, "w": 3, "h": 4},
                                            ),
                                        ) as recognize_frame:
                                            with patch.object(
                                                face, "draw_face_box"
                                            ) as draw_face_box:
                                                with patch.object(
                                                    face.cv2, "imshow"
                                                ), patch.object(
                                                    face.cv2,
                                                    "waitKey",
                                                    side_effect=[0] * 9 + [ord("q")],
                                                ), patch.object(
                                                    face.cv2, "putText"
                                                ) as put_text, patch.object(
                                                    face.cv2, "destroyAllWindows"
                                                ):
                                                    face.run_model_comparison_webcam()

        select_target_bin.assert_called_once()
        self.assertIs(select_target_bin.call_args.args[0], camera)
        recognize_frame.assert_called_once()
        np.testing.assert_array_equal(recognize_frame.call_args.args[0], frame)
        self.assertIs(recognize_frame.call_args.args[1], embeddings)
        self.assertEqual(len(draw_face_box.call_args_list), 20)
        self.assertEqual(len(draw_litter.call_args_list), 20)
        self.assertTrue(
            all(
                [detection for detection in call.args[1] if detection["kind"] == "bin"]
                == [target_bin]
                for call in draw_litter.call_args_list
            )
        )
        self.assertTrue(
            all(call.args[2] for call in draw_face_box.call_args_list[-2:])
        )
        face_labels = [
            call.args[1]
            for call in put_text.call_args_list
            if call.args[1].startswith("Face:")
        ]
        self.assertEqual(
            face_labels[-2:],
            ["Face: Identified: Alice (0.100)", "Face: Identified: Alice (0.100)"],
        )

    def test_load_fine_tuned_bin_model_requires_checkpoint(self):
        with patch.object(
            face, "FINE_TUNED_BIN_MODEL_PATH", Path("missing-best.pt")
        ):
            with self.assertRaises(FileNotFoundError):
                face.load_litter_models(face.MODEL_FINE_TUNED)

    def test_load_fine_tuned_bin_model_keeps_item_detector_pretrained(self):
        bin_model = FakeModel(FakeResult([]))
        bin_model.result.names = {0: "bin"}
        item_model = FakeModel(FakeResult([]))
        with TemporaryDirectory() as temporary_directory:
            checkpoint = Path(temporary_directory) / "best.pt"
            checkpoint.write_bytes(b"test checkpoint")
            with patch.object(face, "FINE_TUNED_BIN_MODEL_PATH", checkpoint):
                with patch.object(
                    face, "YOLO", side_effect=[bin_model, item_model]
                ) as yolo:
                    loaded_item_model, loaded_bin_model = face.load_litter_models(
                        face.MODEL_FINE_TUNED
                    )

        self.assertIs(loaded_item_model, item_model)
        self.assertIs(loaded_bin_model, bin_model)
        self.assertEqual(list(item_model.classes), sorted(face.TARGET_CLASSES))
        self.assertEqual(yolo.call_count, 2)

    def test_load_fine_tuned_bin_model_rejects_checkpoint_without_bin_class(self):
        model = FakeModel(FakeResult([]))
        model.result.names = {0: "bottle", 1: "can", 2: "cup"}
        with TemporaryDirectory() as temporary_directory:
            checkpoint = Path(temporary_directory) / "best.pt"
            checkpoint.write_bytes(b"old three-class checkpoint")
            with patch.object(face, "FINE_TUNED_BIN_MODEL_PATH", checkpoint):
                with patch.object(face, "YOLO", return_value=model):
                    with self.assertRaisesRegex(RuntimeError, "class named 'bin'"):
                        face.load_litter_models(face.MODEL_FINE_TUNED)

    def test_detect_litter_includes_bins_but_marks_them_separately(self):
        result = FakeResult(
            [
                FakeBox(0, 0.9, (10, 20, 50, 80)),
                FakeBox(1, 0.8, (100, 120, 200, 300)),
                FakeBox(2, 0.99, (30, 40, 60, 90)),
                FakeBox(3, 0.2, (210, 310, 260, 380)),
                FakeBox(3, 0.34, (220, 320, 270, 390)),
                FakeBox(3, 0.36, (230, 330, 280, 400)),
            ]
        )

        detections = face.detect_litter(
            FakeModel(result), np.zeros((480, 640, 3)), (0, 0, 640, 480)
        )

        self.assertEqual(
            [(detection["label"], detection["kind"]) for detection in detections],
            [
                ("bottle", "recyclable"),
                ("trash can", "bin"),
                ("recycling bin", "bin"),
            ],
        )
        for detection, expected_confidence in zip(detections, [0.9, 0.8, 0.36]):
            self.assertAlmostEqual(detection["confidence"], expected_confidence, places=6)

    def test_detect_litter_uses_original_yolo_world_confidence_threshold(self):
        frame = np.zeros((480, 640, 3))
        model = MagicMock(
            return_value=[
                FakeResult(
                    [
                        FakeBox(0, 0.34, (10, 20, 50, 80)),
                        FakeBox(0, 0.36, (60, 70, 90, 110)),
                    ]
                )
            ]
        )

        detections = face.detect_litter(model, frame, None)

        model.assert_called_once_with(frame, verbose=False, conf=0.35)
        self.assertEqual(len(detections), 1)
        self.assertEqual(detections[0]["confidence"], 0.36)

    def test_detect_litter_combines_fine_tuned_items_with_pretrained_bins(self):
        item_model = FakeModel(
            FakeResult(
                [
                    FakeBox(0, 0.9, (10, 20, 50, 80)),
                    FakeBox(1, 0.8, (60, 70, 90, 110)),
                ]
            )
        )
        item_model.result.names = {0: "bottle", 1: "can"}
        bin_model = FakeModel(
            FakeResult([FakeBox(0, 0.95, (100, 120, 200, 300))])
        )
        bin_model.result.names = {0: "recycling bin"}

        detections = face.detect_litter(
            item_model,
            np.zeros((480, 640, 3)),
            None,
            bin_model=bin_model,
        )

        self.assertEqual(
            [(detection["label"], detection["kind"]) for detection in detections],
            [
                ("bottle", "recyclable"),
                ("can", "recyclable"),
                ("recycling bin", "bin"),
            ],
        )

    def test_detect_litter_requires_separate_bin_model(self):
        model = FakeModel(
            FakeResult(
                [
                    FakeBox(0, 0.9, (10, 20, 50, 80)),
                    FakeBox(1, 0.85, (100, 120, 200, 300)),
                ]
            )
        )
        model.result.names = {0: "bottle", 1: "bin"}

        detections = face.detect_litter(model, np.zeros((480, 640, 3)), None)

        self.assertEqual(
            [(detection["label"], detection["kind"]) for detection in detections],
            [("bottle", "recyclable")],
        )

    def test_detect_items_uses_lower_threshold_inside_drop_zone(self):
        frame = np.zeros((80, 80, 3))
        model = MagicMock(
            return_value=[
                FakeResult(
                    [
                        FakeBox(0, 0.15, (10, 10, 20, 20)),
                        FakeBox(0, 0.24, (50, 50, 60, 60)),
                        FakeBox(0, 0.25, (60, 60, 70, 70)),
                    ]
                )
            ]
        )

        detections = face.detect_items(model, frame, (5, 5, 30, 30))

        model.assert_called_once_with(frame, verbose=False, conf=0.15)
        self.assertEqual(
            [detection["confidence"] for detection in detections], [0.15, 0.25]
        )

    def test_drop_zone_tracks_bin_opening_in_camera_frame(self):
        frame = np.zeros((480, 640, 3))

        self.assertIsNone(face.get_drop_zone(frame))
        self.assertEqual(
            face.get_drop_zone(frame, {"box": (100, 120, 300, 320)}),
            (110, 80, 290, 200),
        )

    def test_bottle_is_marked_inside_only_when_bin_is_detected(self):
        detections = [
            {"center": (200, 150), "kind": "recyclable"},
            {"center": (500, 300), "kind": "recyclable"},
        ]

        face.mark_detections_in_drop_zone(detections, None)
        self.assertEqual(
            [detection["inside"] for detection in detections], [False, False]
        )

        face.mark_detections_in_drop_zone(detections, (120, 120, 280, 190))
        self.assertEqual(
            [detection["inside"] for detection in detections], [True, False]
        )

    def test_bottle_drop_is_detected_at_dynamic_bin_opening(self):
        tracks = {}
        drop_zone = (120, 120, 280, 190)

        def detection(center):
            return {
                "label": "bottle",
                "center": center,
                "inside": drop_zone[0] <= center[0] <= drop_zone[2]
                and drop_zone[1] <= center[1] <= drop_zone[3],
            }

        self.assertFalse(
            face.update_drop_tracks(tracks, [detection((200, 100))], drop_zone)
        )
        self.assertFalse(
            face.update_drop_tracks(tracks, [detection((200, 125))], drop_zone)
        )
        self.assertFalse(
            face.update_drop_tracks(tracks, [detection((200, 126))], drop_zone)
        )
        self.assertFalse(
            face.update_drop_tracks(tracks, [detection((200, 127))], drop_zone)
        )
        self.assertTrue(
            face.update_drop_tracks(tracks, [detection((200, 128))], drop_zone)
        )

    def test_fast_drop_disappearing_inside_zone_is_confirmed(self):
        frame = np.zeros((480, 640, 3))
        bin_box = (100, 100, 300, 300)
        drop_zone = face.get_drop_zone(frame, {"box": bin_box})
        bin_height = bin_box[3] - bin_box[1]
        tracks = {}
        recent_above = []

        def detection(center):
            return {
                "label": "bottle",
                "center": center,
                "inside": (
                    drop_zone[0] <= center[0] <= drop_zone[2]
                    and drop_zone[1] <= center[1] <= drop_zone[3]
                ),
            }

        self.assertFalse(
            face.update_drop_tracks(
                tracks, [detection((200, 50))], drop_zone, bin_height, recent_above
            )
        )
        self.assertFalse(
            face.update_drop_tracks(
                tracks, [detection((200, 70))], drop_zone, bin_height, recent_above
            )
        )
        for _ in range(face.DROP_MISSING_FRAMES - 1):
            self.assertFalse(
                face.update_drop_tracks(
                    tracks, [], drop_zone, bin_height, recent_above
                )
            )
        self.assertTrue(
            face.update_drop_tracks(
                tracks, [], drop_zone, bin_height, recent_above
            )
        )

    def test_slow_drop_is_confirmed_after_settling(self):
        frame = np.zeros((480, 640, 3))
        bin_box = (100, 100, 300, 300)
        drop_zone = face.get_drop_zone(frame, {"box": bin_box})
        bin_height = bin_box[3] - bin_box[1]
        tracks = {}
        recent_above = []

        def detection(center):
            return {
                "label": "bottle",
                "center": center,
                "inside": (
                    drop_zone[0] <= center[0] <= drop_zone[2]
                    and drop_zone[1] <= center[1] <= drop_zone[3]
                ),
            }

        for center in ((200, 50), (200, 70), (200, 71), (200, 72)):
            self.assertFalse(
                face.update_drop_tracks(
                    tracks, [detection(center)], drop_zone, bin_height, recent_above
                )
            )
        self.assertTrue(
            face.update_drop_tracks(
                tracks,
                [detection((200, 73))],
                drop_zone,
                bin_height,
                recent_above,
            )
        )

    def test_bottle_held_above_zone_does_not_count_as_drop(self):
        frame = np.zeros((480, 640, 3))
        bin_box = (100, 100, 300, 300)
        drop_zone = face.get_drop_zone(frame, {"box": bin_box})
        bin_height = bin_box[3] - bin_box[1]
        tracks = {}
        recent_above = []
        held_detection = {
            "label": "bottle",
            "center": (200, 50),
            "inside": False,
        }

        for _ in range(5):
            self.assertFalse(
                face.update_drop_tracks(
                    tracks,
                    [held_detection],
                    drop_zone,
                    bin_height,
                    recent_above,
                )
            )
        for _ in range(face.DROP_MISSING_FRAMES + 1):
            self.assertFalse(
                face.update_drop_tracks(
                    tracks, [], drop_zone, bin_height, recent_above
                )
            )

    def test_new_inside_track_uses_recent_above_detection(self):
        frame = np.zeros((480, 640, 3))
        bin_box = (100, 100, 300, 300)
        drop_zone = face.get_drop_zone(frame, {"box": bin_box})
        bin_height = bin_box[3] - bin_box[1]
        tracks = {}
        recent_above = []

        def detection(center):
            return {
                "label": "bottle",
                "center": center,
                "inside": (
                    drop_zone[0] <= center[0] <= drop_zone[2]
                    and drop_zone[1] <= center[1] <= drop_zone[3]
                ),
            }

        face.update_drop_tracks(
            tracks, [detection((200, 50))], drop_zone, bin_height, recent_above
        )
        tracks.clear()
        self.assertFalse(
            face.update_drop_tracks(
                tracks,
                [detection((200, 70))],
                drop_zone,
                bin_height,
                recent_above,
            )
        )
        self.assertTrue(tracks[0]["drop_candidate"])
        for _ in range(face.DROP_MISSING_FRAMES - 1):
            self.assertFalse(
                face.update_drop_tracks(
                    tracks, [], drop_zone, bin_height, recent_above
                )
            )
        self.assertTrue(
            face.update_drop_tracks(
                tracks, [], drop_zone, bin_height, recent_above
            )
        )


if __name__ == "__main__":
    unittest.main()
