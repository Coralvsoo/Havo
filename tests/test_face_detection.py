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

    def __call__(self, _frame, verbose=False):
        return [self.result]


class FaceDetectionTests(unittest.TestCase):
    def test_detect_litter_includes_bins_but_marks_them_separately(self):
        result = FakeResult(
            [
                FakeBox(0, 0.9, (10, 20, 50, 80)),
                FakeBox(1, 0.8, (100, 120, 200, 300)),
                FakeBox(2, 0.99, (30, 40, 60, 90)),
                FakeBox(3, 0.2, (210, 310, 260, 380)),
            ]
        )

        detections = face.detect_litter(
            FakeModel(result), np.zeros((480, 640, 3)), (0, 0, 640, 480)
        )

        self.assertEqual(
            [(detection["label"], detection["kind"]) for detection in detections],
            [("bottle", "recyclable"), ("trash can", "bin")],
        )


if __name__ == "__main__":
    unittest.main()
