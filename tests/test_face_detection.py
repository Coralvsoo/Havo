import unittest

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

    def __call__(self, _frame, verbose=False, conf=None):
        return [self.result]


class FaceDetectionTests(unittest.TestCase):
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
