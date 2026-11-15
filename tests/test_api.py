import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np
from fastapi.testclient import TestClient

import api
import face


class HavoApiTests(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary_directory.cleanup)
        self.database_patch = patch.object(
            face,
            "DATABASE_PATH",
            Path(self.temporary_directory.name) / "test.db",
        )
        self.database_patch.start()
        self.addCleanup(self.database_patch.stop)
        self.environment_patch = patch.dict(os.environ, {"HAVO_API_KEY": "test-key"})
        self.environment_patch.start()
        self.addCleanup(self.environment_patch.stop)
        api.sessions.clear()
        api.detector_models = None
        self.client = TestClient(api.app)
        self.headers = {"X-API-Key": "test-key"}

    def jpeg_upload(self):
        success, encoded = cv2.imencode(
            ".jpg", np.zeros((80, 100, 3), dtype=np.uint8)
        )
        self.assertTrue(success)
        return ("frame.jpg", encoded.tobytes(), "image/jpeg")

    def test_health_is_public(self):
        response = self.client.get("/health")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"status": "ok"})

    def test_protected_routes_require_api_key(self):
        response = self.client.get("/api/v1/users/Alice/points")

        self.assertEqual(response.status_code, 401)

    def test_protected_routes_fail_closed_without_configured_key(self):
        with patch.dict(os.environ, {}, clear=True):
            response = self.client.get(
                "/api/v1/users/Alice/points",
                headers={"X-API-Key": "any-value"},
            )

        self.assertEqual(response.status_code, 503)

    def test_register_user_and_get_points(self):
        with patch.object(
            face, "extract_embedding", return_value=np.array([0.1, 0.2])
        ):
            response = self.client.post(
                "/api/v1/users/register",
                headers=self.headers,
                data={"user_name": " Alice "},
                files={"image": self.jpeg_upload()},
            )

        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.json()["registered"])
        self.assertEqual(response.json()["user_name"], "Alice")
        self.assertEqual(
            response.json()["note"],
            "The uploaded image was not stored; only its face embedding was saved.",
        )

        points = self.client.get(
            "/api/v1/users/Alice/points", headers=self.headers
        )
        self.assertEqual(points.status_code, 200)
        self.assertEqual(points.json(), {"user_name": "Alice", "points": 0})

    def test_process_frame_returns_detections_and_awards_drop_points(self):
        with patch.object(
            face, "extract_embedding", return_value=np.array([0.1, 0.2])
        ):
            registration = self.client.post(
                "/api/v1/users/register",
                headers=self.headers,
                data={"user_name": "Alice"},
                files={"image": self.jpeg_upload()},
            )
        self.assertEqual(registration.status_code, 200)

        api.detector_models = (object(), None)
        detections = [
            {
                "box": (20, 10, 80, 70),
                "label": "recycling bin",
                "confidence": 0.9,
                "kind": "bin",
                "center": (50, 40),
                "inside": False,
            },
            {
                "box": (40, 20, 60, 40),
                "label": "bottle",
                "confidence": 0.9,
                "kind": "recyclable",
                "center": (50, 30),
                "inside": False,
            },
        ]
        with patch.object(face, "detect_litter", return_value=detections):
            with patch.object(
                face,
                "recognize_frame",
                return_value=("Identified: Alice", 0.1, {"x": 1, "y": 2, "w": 3, "h": 4}),
            ):
                with patch.object(
                    face, "update_drop_tracks", side_effect=[True, True]
                ):
                    response = self.client.post(
                        "/api/v1/frames/process",
                        headers=self.headers,
                        data={"session_id": "camera-1"},
                        files={"image": self.jpeg_upload()},
                    )
                    cooldown_response = self.client.post(
                        "/api/v1/frames/process",
                        headers=self.headers,
                        data={"session_id": "camera-1"},
                        files={"image": self.jpeg_upload()},
                    )

        self.assertEqual(response.status_code, 200, response.text)
        result = response.json()
        self.assertEqual(result["face"]["user_name"], "Alice")
        self.assertEqual(result["selected_bin"]["label"], "recycling bin")
        self.assertEqual(len(result["detections"]), 2)
        self.assertTrue(result["drop_detected"])
        self.assertTrue(result["points_awarded"])
        self.assertEqual(result["points"], 10)
        self.assertEqual(cooldown_response.status_code, 200)
        self.assertFalse(cooldown_response.json()["points_awarded"])
        self.assertEqual(cooldown_response.json()["points"], 10)

    def test_invalid_image_is_rejected(self):
        response = self.client.post(
            "/api/v1/frames/process",
            headers=self.headers,
            data={"session_id": "camera-1"},
            files={"image": ("frame.jpg", b"not an image", "image/jpeg")},
        )

        self.assertEqual(response.status_code, 415)

    def test_docs_expose_api_key_scheme(self):
        schema = self.client.get("/openapi.json").json()

        self.assertEqual(
            schema["components"]["securitySchemes"]["APIKeyHeader"]["name"],
            "X-API-Key",
        )
        self.assertIn(
            {"APIKeyHeader": []},
            schema["paths"]["/api/v1/frames/process"]["post"]["security"],
        )


if __name__ == "__main__":
    unittest.main()
