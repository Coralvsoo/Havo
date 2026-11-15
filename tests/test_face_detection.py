import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import MagicMock, patch

import numpy as np

import face


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


class FaceDetectionTests(unittest.TestCase):
    def test_load_old_model_sets_existing_yolo_world_classes(self):
        model = FakeModel(FakeResult([]))

        with patch.object(face, "YOLO", return_value=model) as yolo:
            item_model, bin_model = face.load_litter_models(face.MODEL_OLD)

        self.assertIs(item_model, model)
        self.assertIsNone(bin_model)
        yolo.assert_called_once_with("yolov8s-world.pt")
        self.assertEqual(list(model.classes), list(face.YOLO_CLASSES))

    def test_select_target_bin_uses_user_selected_camera_region(self):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        with patch.object(face.cv2, "selectROI", return_value=(100, 120, 200, 240)):
            target_bin = face.select_target_bin(frame)

        self.assertEqual(target_bin["box"], (100, 120, 300, 360))
        self.assertEqual(target_bin["center"], (200, 240))
        self.assertEqual(target_bin["label"], "selected bin")
        self.assertEqual(target_bin["kind"], "bin")

    def test_select_target_bin_requires_a_nonempty_selection(self):
        with patch.object(face.cv2, "selectROI", return_value=(0, 0, 0, 0)):
            with self.assertRaisesRegex(RuntimeError, "No target bin was selected"):
                face.select_target_bin(np.zeros((480, 640, 3), dtype=np.uint8))

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
        np.testing.assert_array_equal(select_target_bin.call_args.args[0], frame)
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
                FakeBox(3, 0.3, (220, 320, 270, 390)),
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
        for detection, expected_confidence in zip(detections, [0.9, 0.8, 0.3]):
            self.assertAlmostEqual(detection["confidence"], expected_confidence, places=6)

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

    def test_drop_zone_tracks_bin_opening_in_camera_frame(self):
        frame = np.zeros((480, 640, 3))

        self.assertIsNone(face.get_drop_zone(frame))
        self.assertEqual(
            face.get_drop_zone(frame, {"box": (100, 120, 300, 320)}),
            (120, 120, 280, 190),
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


if __name__ == "__main__":
    unittest.main()
