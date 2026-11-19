import unittest

import face


class FaceIdentityTests(unittest.TestCase):
    def test_match_sets_user(self):
        state = face.update_identity(
            face.new_identity_state(), "Identified: Ali", 0.0
        )

        self.assertEqual(state["user"], "Ali")
        self.assertEqual(state["misses"], 0)

    def test_no_face_found_keeps_user(self):
        state = face.update_identity(
            face.new_identity_state(), "Identified: Ali", 0.0
        )
        state = face.update_identity(state, "Scanning error", 1.0)

        self.assertEqual(state["user"], "Ali")
        self.assertEqual(state["misses"], 1)

    def test_unknown_face_clears_user_immediately(self):
        state = face.update_identity(
            face.new_identity_state(), "Identified: Ali", 0.0
        )
        state = face.update_identity(state, "Unknown User", 0.5, distance=0.62)

        self.assertIsNone(state["user"])

    def test_three_misses_clear_user(self):
        state = face.update_identity(
            face.new_identity_state(), "Identified: Ali", 0.0
        )
        for now in (0.5, 1.0, 1.5):
            state = face.update_identity(state, "Scanning error", now)

        self.assertIsNone(state["user"])

    def test_match_resets_miss_counter(self):
        state = face.update_identity(
            face.new_identity_state(), "Identified: Ali", 0.0
        )
        state = face.update_identity(state, "Scanning error", 0.5)
        state = face.update_identity(state, "Scanning error", 1.0)
        state = face.update_identity(state, "Identified: Ali", 1.5)
        state = face.update_identity(state, "Scanning error", 2.0)

        self.assertEqual(state["user"], "Ali")
        self.assertEqual(state["misses"], 1)

    def test_hold_expires_after_timeout(self):
        state = face.update_identity(
            face.new_identity_state(), "Identified: Ali", 0.0
        )
        state = face.expire_identity(state, face.FACE_HOLD_SECONDS + 0.1)

        self.assertIsNone(state["user"])

    def test_new_user_replaces_old_user(self):
        state = face.update_identity(
            face.new_identity_state(), "Identified: Ali", 0.0
        )
        state = face.update_identity(state, "Identified: Vali", 1.0)

        self.assertEqual(state["user"], "Vali")


if __name__ == "__main__":
    unittest.main()
