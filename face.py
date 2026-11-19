import argparse
from contextlib import closing
import csv
import json
import os
import sqlite3
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

PROJECT_DIR = Path(__file__).resolve().parent
LOCAL_DEEPFACE_SOURCE = PROJECT_DIR / "deepface"
if (LOCAL_DEEPFACE_SOURCE / "deepface" / "__init__.py").is_file():
    sys.path.insert(0, str(LOCAL_DEEPFACE_SOURCE))

from deepface import DeepFace

from hardware.esp32_serial import SerialSensorSource
from hardware.mock_sensor import MockSensorSource
from session_recording import SessionRecorder


DATABASE_PATH = Path(
    os.environ.get("HAVO_DATABASE_PATH", PROJECT_DIR / "smartclean_app.db")
)
MODEL_NAME = "Facenet"
FINE_TUNED_BIN_MODEL_PATHS = (
    PROJECT_DIR / "P04_bin_dataset2_best.pt",
    Path.home() / "Downloads" / "P04_bin_dataset2_best.pt",
    PROJECT_DIR / "weights" / "p04_bin" / "best.pt",
)
FINE_TUNED_BIN_MODEL_PATH = next(
    (path for path in FINE_TUNED_BIN_MODEL_PATHS if path.is_file()),
    FINE_TUNED_BIN_MODEL_PATHS[-1],
)
MATCH_THRESHOLD = 0.40
POINTS_PER_DROP = 10
DROP_COOLDOWN_SECONDS = 3.0
YOLO_FRAME_INTERVAL = 1
DETECTION_CONFIDENCE = 0.35
BIN_CONFIDENCE = 0.49
CALIBRATION_FRAMES = 30
CALIBRATION_MIN_HITS = 20
CALIBRATION_MIN_IOU = 0.70
DRIFT_CHECK_SECONDS = 5.0
DRIFT_IOU_THRESHOLD = 0.50
DRIFT_CONSECUTIVE_CHECKS = 3
ZONE_LOCK = "lock"
ZONE_DETECT = "detect"
ZONE_FIXED = "fixed"
CLIP_SECONDS_BEFORE = 3.0
CLIP_SECONDS_AFTER = 3.0
SEGMENT_MINUTES = 5
ZONE_ACTIVITY_TIMEOUT = 2.0
MATCH_TOLERANCE_SECONDS = 1.0
DEFAULT_MAX_STORAGE_GB = 5.0
TARGET_CLASSES = {"bottle", "cup", "can"}
BIN_CLASSES = {"trash can", "recycling bin"}
YOLO_CLASSES = ("bottle", "cup", "can", "trash can", "recycling bin")
MODEL_OLD = "old"
MODEL_FINE_TUNED = "fine_tuned"
MODEL_COMPARE = "compare"
TRACK_MATCH_DISTANCE_RATIO = 0.5
DROP_MIN_DOWNWARD_RATIO = 0.05
DROP_SETTLE_DISTANCE_RATIO = 0.08
DROP_ZONE_TOP_RATIO = 0.20
DROP_ZONE_BOTTOM_RATIO = 0.40
DROP_ZONE_SIDE_INSET_RATIO = 0.05
RECYCLABLE_ZONE_CONFIDENCE = 0.15
RECYCLABLE_OUTSIDE_ZONE_CONFIDENCE = 0.25
DROP_MISSING_FRAMES = 4
TRACK_MAX_MISSING_FRAMES = 4
MIN_SETTLED_FRAMES = 3


def connect_database():
    try:
        connection = sqlite3.connect(DATABASE_PATH)
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS face_embeddings (
                id INTEGER PRIMARY KEY,
                user_name TEXT UNIQUE NOT NULL,
                vector_json TEXT NOT NULL
            )
            """
        )
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS user_points (
                user_name TEXT PRIMARY KEY,
                points INTEGER NOT NULL DEFAULT 0,
                FOREIGN KEY (user_name) REFERENCES face_embeddings(user_name)
            )
            """
        )
        connection.commit()
        return connection
    except sqlite3.Error as error:
        raise RuntimeError(f"Could not open face database: {error}") from error


def extract_embedding(image):
    representations = DeepFace.represent(image, model_name=MODEL_NAME)
    if not representations or "embedding" not in representations[0]:
        raise ValueError("No face embedding was found in the image")
    return np.asarray(representations[0]["embedding"], dtype=np.float32)


def save_user_embedding(name, embedding):
    clean_name = name.strip()
    if not clean_name:
        raise ValueError("User name cannot be empty")

    vector_json = json.dumps(np.asarray(embedding, dtype=np.float32).tolist())
    try:
        with closing(connect_database()) as connection:
            connection.execute(
                """
                INSERT INTO face_embeddings (user_name, vector_json)
                VALUES (?, ?)
                ON CONFLICT(user_name) DO UPDATE SET vector_json = excluded.vector_json
                """,
                (clean_name, vector_json),
            )
            connection.commit()
    except sqlite3.Error as error:
        raise RuntimeError(f"Could not save face embedding: {error}") from error
    return clean_name


def add_user(name, image_path):
    clean_name = name.strip()
    if not clean_name:
        raise ValueError("User name cannot be empty")

    source = Path(image_path).resolve()
    if not source.is_file():
        raise FileNotFoundError(f"Image not found: {source}")

    try:
        embedding = extract_embedding(str(source))
        save_user_embedding(clean_name, embedding)
        print(f"Registered {clean_name} in {DATABASE_PATH}")
    except (OSError, RuntimeError, ValueError) as error:
        raise RuntimeError(f"Could not register {clean_name}: {error}") from error


def load_embeddings():
    try:
        with closing(connect_database()) as connection:
            rows = connection.execute(
                "SELECT user_name, vector_json FROM face_embeddings"
            ).fetchall()
    except sqlite3.Error as error:
        raise RuntimeError(f"Could not load face embeddings: {error}") from error

    embeddings = []
    for user_name, vector_json in rows:
        try:
            vector = np.asarray(json.loads(vector_json), dtype=np.float32)
            if vector.ndim != 1 or vector.size == 0:
                raise ValueError("embedding must be a non-empty vector")
            embeddings.append((user_name, vector))
        except (json.JSONDecodeError, TypeError, ValueError) as error:
            print(f"Skipping invalid embedding for {user_name}: {error}", file=sys.stderr)
    return embeddings


def award_points(user_name):
    if not user_name:
        return None

    try:
        with closing(connect_database()) as connection:
            registered = connection.execute(
                "SELECT 1 FROM face_embeddings WHERE user_name = ?",
                (user_name,),
            ).fetchone()
            if not registered:
                raise ValueError(f"User is not registered: {user_name}")
            connection.execute(
                """
                INSERT INTO user_points (user_name, points)
                VALUES (?, ?)
                ON CONFLICT(user_name) DO UPDATE SET points = points + excluded.points
                """,
                (user_name, POINTS_PER_DROP),
            )
            total = connection.execute(
                "SELECT points FROM user_points WHERE user_name = ?",
                (user_name,),
            ).fetchone()[0]
            connection.commit()
            return total
    except sqlite3.Error as error:
        raise RuntimeError(f"Could not award points: {error}") from error


def get_user_points(user_name):
    if not user_name:
        return 0
    try:
        with closing(connect_database()) as connection:
            row = connection.execute(
                "SELECT points FROM user_points WHERE user_name = ?",
                (user_name,),
            ).fetchone()
        return row[0] if row else 0
    except sqlite3.Error as error:
        raise RuntimeError(f"Could not load user points: {error}") from error


def cosine_distance(first, second):
    first_norm = np.linalg.norm(first)
    second_norm = np.linalg.norm(second)
    if first_norm == 0 or second_norm == 0:
        return 1.0
    similarity = np.dot(first, second) / (first_norm * second_norm)
    return float(1.0 - np.clip(similarity, -1.0, 1.0))


def recognize_frame(frame, known_embeddings):
    if not known_embeddings:
        return "No users registered", None, None

    try:
        representations = DeepFace.represent(frame, model_name=MODEL_NAME)
        if not representations or "embedding" not in representations[0]:
            return "Unknown User", None, None

        live_embedding = np.asarray(
            representations[0]["embedding"], dtype=np.float32
        )
        matches = [
            (cosine_distance(live_embedding, saved_vector), user_name)
            for user_name, saved_vector in known_embeddings
            if saved_vector.shape == live_embedding.shape
        ]
        facial_area = representations[0].get("facial_area")
        if not matches:
            return "Unknown User", None, facial_area

        distance, user_name = min(matches)
        if distance <= MATCH_THRESHOLD:
            return f"Identified: {user_name}", distance, facial_area
        return "Unknown User", distance, facial_area
    except Exception:
        print("Face recognition error:", file=sys.stderr)
        import traceback

        traceback.print_exc()
        return "Scanning error", None, None


def draw_face_box(frame, facial_area, identified):
    if not facial_area:
        return
    x = facial_area.get("x", 0)
    y = facial_area.get("y", 0)
    width = facial_area.get("w", 0)
    height = facial_area.get("h", 0)
    color = (0, 255, 0) if identified else (0, 0, 255)
    cv2.rectangle(frame, (x, y), (x + width, y + height), color, 2)


def get_drop_zone(frame, bin_detection=None):
    if bin_detection is None:
        return None

    height, width = frame.shape[:2]
    x1, y1, x2, y2 = bin_detection["box"]
    bin_width = x2 - x1
    bin_height = y2 - y1
    return (
        max(0, min(width - 1, int(x1 + bin_width * DROP_ZONE_SIDE_INSET_RATIO))),
        max(0, min(height - 1, int(y1 - bin_height * DROP_ZONE_TOP_RATIO))),
        max(1, min(width, int(x2 - bin_width * DROP_ZONE_SIDE_INSET_RATIO))),
        max(
            1,
            min(height, int(y1 + bin_height * DROP_ZONE_BOTTOM_RATIO)),
        ),
    )


def box_iou(first_box, second_box):
    first_x1, first_y1, first_x2, first_y2 = first_box
    second_x1, second_y1, second_x2, second_y2 = second_box
    intersection_width = max(
        0, min(first_x2, second_x2) - max(first_x1, second_x1)
    )
    intersection_height = max(
        0, min(first_y2, second_y2) - max(first_y1, second_y1)
    )
    intersection = intersection_width * intersection_height
    first_area = max(0, first_x2 - first_x1) * max(0, first_y2 - first_y1)
    second_area = max(0, second_x2 - second_x1) * max(0, second_y2 - second_y1)
    union = first_area + second_area - intersection
    return intersection / union if union else 0.0


def median_box(boxes):
    if not boxes:
        raise ValueError("At least one box is required")
    return tuple(int(value) for value in np.median(np.asarray(boxes), axis=0))


def decide_calibration(per_frame_boxes):
    boxes = []
    for frame_boxes in per_frame_boxes:
        if frame_boxes is None:
            continue
        if len(frame_boxes) == 4 and all(
            isinstance(value, (int, float, np.integer, np.floating))
            for value in frame_boxes
        ):
            boxes.append(tuple(frame_boxes))
        else:
            boxes.extend(tuple(box) for box in frame_boxes if box is not None)
    if len(boxes) < CALIBRATION_MIN_HITS:
        return None, "too few detections"

    stable_box = median_box(boxes)
    median_iou = float(np.median([box_iou(box, stable_box) for box in boxes]))
    if median_iou < CALIBRATION_MIN_IOU:
        return None, "unstable box"
    return stable_box, "ok"


def update_drift(state, iou_or_none):
    if state.get("drifted"):
        return {"bad_checks": state.get("bad_checks", 0), "drifted": True}
    bad_checks = (
        state.get("bad_checks", 0) + 1
        if iou_or_none is None or iou_or_none < DRIFT_IOU_THRESHOLD
        else 0
    )
    return {
        "bad_checks": bad_checks,
        "drifted": bad_checks >= DRIFT_CONSECUTIVE_CHECKS,
    }


def _run_detector(detector, frame, confidence):
    result = detector(frame, verbose=False, conf=confidence)[0]
    detections = []
    for box in result.boxes:
        class_id = int(box.cls[0].item())
        label = result.names[class_id]
        confidence_value = float(box.conf[0].item())
        x1, y1, x2, y2 = [int(value) for value in box.xyxy[0].tolist()]
        center = ((x1 + x2) // 2, (y1 + y2) // 2)
        detections.append(
            {
                "box": (x1, y1, x2, y2),
                "label": label,
                "confidence": confidence_value,
                "kind": (
                    "bin"
                    if label in BIN_CLASSES or label == "bin"
                    else "recyclable"
                ),
                "center": center,
                "inside": False,
            }
        )
    return detections


def detect_bins(bin_model, frame, conf=BIN_CONFIDENCE):
    return [
        detection
        for detection in _run_detector(bin_model, frame, conf)
        if detection["kind"] == "bin"
    ]


def detect_items(item_model, frame, drop_zone=None):
    detections = []
    for detection in _run_detector(
        item_model, frame, RECYCLABLE_ZONE_CONFIDENCE
    ):
        if (
            detection["kind"] != "recyclable"
            or detection["label"] not in TARGET_CLASSES
        ):
            continue
        center_x, center_y = detection["center"]
        inside_zone = drop_zone is not None and (
            drop_zone[0] <= center_x <= drop_zone[2]
            and drop_zone[1] <= center_y <= drop_zone[3]
        )
        minimum_confidence = (
            RECYCLABLE_ZONE_CONFIDENCE
            if inside_zone
            else RECYCLABLE_OUTSIDE_ZONE_CONFIDENCE
        )
        if detection["confidence"] >= minimum_confidence:
            detections.append(detection)
    return detections


def select_target_bin(camera, frame_provider=None):
    window_name = "Select target bin - drag a box, then press Enter"
    selection = {"start": None, "end": None, "dragging": False}

    def update_selection(event, x, y, _flags, _parameter):
        if event == cv2.EVENT_LBUTTONDOWN:
            selection["start"] = (x, y)
            selection["end"] = (x, y)
            selection["dragging"] = True
        elif event == cv2.EVENT_MOUSEMOVE and selection["dragging"]:
            selection["end"] = (x, y)
        elif event == cv2.EVENT_LBUTTONUP and selection["dragging"]:
            selection["end"] = (x, y)
            selection["dragging"] = False

    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, update_selection)
    try:
        while True:
            if frame_provider is None:
                success, frame = camera.read()
                if not success:
                    raise RuntimeError(
                        "Could not read a frame while selecting the target bin"
                    )
            else:
                provided_frame = frame_provider()
                if provided_frame is None:
                    raise RuntimeError(
                        "Could not read a frame while selecting the target bin"
                    )
                frame = provided_frame[0]

            preview = frame.copy()
            cv2.putText(
                preview,
                "Drag around the bin, then press Enter | Esc to cancel",
                (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 255),
                2,
            )
            start = selection["start"]
            end = selection["end"]
            if start is not None and end is not None:
                cv2.rectangle(preview, start, end, (0, 255, 0), 2)

            cv2.imshow(window_name, preview)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("\r"), 10):
                if not selection["start"] or not selection["end"]:
                    continue
                x1 = min(selection["start"][0], selection["end"][0])
                y1 = min(selection["start"][1], selection["end"][1])
                x2 = max(selection["start"][0], selection["end"][0])
                y2 = max(selection["start"][1], selection["end"][1])
                if x2 <= x1 or y2 <= y1:
                    continue
                center = ((x1 + x2) // 2, (y1 + y2) // 2)
                return {
                    "box": (x1, y1, x2, y2),
                    "label": "selected bin",
                    "confidence": 1.0,
                    "kind": "bin",
                    "center": center,
                    "inside": False,
                }
            if key in (ord("q"), 27):
                raise RuntimeError("Target-bin selection was cancelled")
    finally:
        cv2.destroyWindow(window_name)


def calibrate_bin(
    camera,
    bin_model,
    *,
    frame_provider=None,
    initial_frame=None,
    frame_count=CALIBRATION_FRAMES,
):
    per_frame_boxes = []
    last_frame = initial_frame
    last_frame_info = None
    for index in range(frame_count):
        if index == 0 and initial_frame is not None:
            frame = initial_frame
        elif frame_provider is not None:
            provided_frame = frame_provider()
            if provided_frame is None:
                raise EOFError("Video ended during bin calibration")
            frame, last_frame_info = provided_frame
        else:
            success, frame = camera.read()
            if not success:
                raise RuntimeError("Could not read a frame during bin calibration")
        last_frame = frame
        per_frame_boxes.append(
            [detection["box"] for detection in detect_bins(bin_model, frame)]
        )
    calibrated_box, reason = decide_calibration(per_frame_boxes)
    return calibrated_box, reason, last_frame, last_frame_info


def confirm_bin(camera, frame, box, no_confirm=False, frame_provider=None):
    if no_confirm:
        return {
            "box": tuple(box),
            "label": "calibrated bin",
            "confidence": 1.0,
            "kind": "bin",
            "center": (
                (box[0] + box[2]) // 2,
                (box[1] + box[3]) // 2,
            ),
            "inside": False,
        }

    window_name = "Confirm calibrated bin"
    cv2.namedWindow(window_name)
    try:
        while True:
            preview = frame.copy()
            cv2.rectangle(preview, box[:2], box[2:], (0, 255, 0), 2)
            cv2.putText(
                preview,
                "Enter/Space accept | R redraw | Esc manual selection",
                (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 255),
                2,
            )
            cv2.imshow(window_name, preview)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("\r"), 10, ord(" ")):
                break
            if key in (ord("r"), 27):
                return select_target_bin(camera, frame_provider=frame_provider)
    finally:
        cv2.destroyWindow(window_name)
    return {
        "box": tuple(box),
        "label": "calibrated bin",
        "confidence": 1.0,
        "kind": "bin",
        "center": (
            (box[0] + box[2]) // 2,
            (box[1] + box[3]) // 2,
        ),
        "inside": False,
    }


def keep_selected_bin(detections, target_bin):
    selected_detections = [
        detection for detection in detections if detection["kind"] == "recyclable"
    ]
    selected_detections.append(dict(target_bin))
    return selected_detections


def get_detected_bin(detections):
    bins = [detection for detection in detections if detection["kind"] == "bin"]
    return next(
        (
            detection
            for detection in bins
            if detection["label"] == "recycling bin"
        ),
        bins[0] if bins else None,
    )


def detect_litter(model, frame, drop_zone, bin_model=None):
    detections = []
    model_classes = [(model, TARGET_CLASSES | BIN_CLASSES)]
    if bin_model is not None:
        model_classes = [
            (model, TARGET_CLASSES),
            (bin_model, {"bin"} | BIN_CLASSES),
        ]

    for detector, accepted_classes in model_classes:
        result = detector(
            frame, verbose=False, conf=DETECTION_CONFIDENCE
        )[0]
        for box in result.boxes:
            class_id = int(box.cls[0].item())
            label = result.names[class_id]
            if label not in accepted_classes:
                continue
            if float(box.conf[0].item()) < DETECTION_CONFIDENCE:
                continue

            x1, y1, x2, y2 = [int(value) for value in box.xyxy[0].tolist()]
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2
            inside = False
            if drop_zone is not None:
                x1_zone, y1_zone, x2_zone, y2_zone = drop_zone
                inside = (
                    x1_zone <= center_x <= x2_zone
                    and y1_zone <= center_y <= y2_zone
                )
            detections.append(
                {
                    "box": (x1, y1, x2, y2),
                    "label": label,
                    "confidence": float(box.conf[0].item()),
                    "kind": (
                        "bin"
                        if label in BIN_CLASSES or label == "bin"
                        else "recyclable"
                    ),
                    "center": (center_x, center_y),
                    "inside": inside,
                }
            )

    return detections


def mark_detections_in_drop_zone(detections, drop_zone):
    for detection in detections:
        inside = False
        if drop_zone is not None:
            x1_zone, y1_zone, x2_zone, y2_zone = drop_zone
            center_x, center_y = detection["center"]
            inside = (
                x1_zone <= center_x <= x2_zone
                and y1_zone <= center_y <= y2_zone
            )
        detection["inside"] = inside


def update_drop_tracks(
    tracks, detections, drop_zone, bin_height=None, recent_above=None
):
    if drop_zone is None:
        if recent_above is not None:
            recent_above.clear()
        return False

    if recent_above is None:
        recent_above = []
    else:
        for previous_detection in recent_above:
            previous_detection["age"] += 1
        recent_above[:] = [
            previous_detection
            for previous_detection in recent_above
            if previous_detection["age"] <= 3
        ]

    matched_track_ids = set()
    drop_ready = False
    zone_top = drop_zone[1]
    if bin_height is None:
        bin_height = (drop_zone[3] - drop_zone[1]) / (
            DROP_ZONE_TOP_RATIO + DROP_ZONE_BOTTOM_RATIO
        )
    bin_height = max(1, bin_height)
    match_distance = bin_height * TRACK_MATCH_DISTANCE_RATIO
    min_downward_move = bin_height * DROP_MIN_DOWNWARD_RATIO
    settle_distance = bin_height * DROP_SETTLE_DISTANCE_RATIO

    for detection in detections:
        center_x, center_y = detection["center"]
        best_track = None
        best_distance = match_distance
        for track_id, track in tracks.items():
            if track_id in matched_track_ids or track["label"] != detection["label"]:
                continue
            distance = np.hypot(
                center_x - track["center"][0], center_y - track["center"][1]
            )
            if distance < best_distance:
                best_track = track_id
                best_distance = distance

        entered_from_history = False
        if best_track is None:
            previous_detection = None
            if detection["inside"]:
                history_matches = []
                for candidate in recent_above:
                    if (
                        candidate["label"] != detection["label"]
                        or candidate["center"][1] >= zone_top
                    ):
                        continue
                    distance = np.hypot(
                        center_x - candidate["center"][0],
                        center_y - candidate["center"][1],
                    )
                    if (
                        distance < match_distance
                        and center_y - candidate["center"][1]
                        >= min_downward_move
                    ):
                        history_matches.append((candidate, distance))
                if history_matches:
                    previous_detection = max(
                        history_matches,
                        key=lambda candidate_and_distance: (
                            candidate_and_distance[0]["age"],
                            -candidate_and_distance[1],
                        ),
                    )[0]
                    recent_above.remove(previous_detection)
                    entered_from_history = True

            best_track = max(tracks.keys(), default=-1) + 1
            previous_center = (
                previous_detection["center"]
                if previous_detection is not None
                else detection["center"]
            )
            tracks[best_track] = {
                "label": detection["label"],
                "center": previous_center,
                "was_inside": False,
                "drop_candidate": entered_from_history,
                "settled_frames": 0,
                "drop_confirmed": False,
                "missing_frames": 0,
            }

        track = tracks[best_track]
        previous_x, previous_y = track["center"]
        center_x, center_y = detection["center"]
        track["missing_frames"] = 0
        if detection["inside"]:
            if (
                not track["was_inside"]
                and previous_y < zone_top
                and center_y - previous_y >= min_downward_move
            ):
                track["drop_candidate"] = True
                track["settled_frames"] = 0
            elif track["drop_candidate"] and not track["drop_confirmed"]:
                movement = np.hypot(center_x - previous_x, center_y - previous_y)
                if movement <= settle_distance:
                    track["settled_frames"] += 1
                    if track["settled_frames"] >= MIN_SETTLED_FRAMES:
                        track["drop_confirmed"] = True
                        drop_ready = True
                else:
                    track["settled_frames"] = 0
            track["was_inside"] = True
        else:
            track["was_inside"] = False
            track["drop_candidate"] = False
            track["settled_frames"] = 0
        track["center"] = detection["center"]
        matched_track_ids.add(best_track)

    for track_id in list(tracks):
        if track_id in matched_track_ids:
            continue
        track = tracks[track_id]
        track["missing_frames"] += 1
        if (
            track["drop_candidate"]
            and not track["drop_confirmed"]
            and track["missing_frames"] >= DROP_MISSING_FRAMES
        ):
            track["drop_confirmed"] = True
            drop_ready = True
        if track["missing_frames"] > TRACK_MAX_MISSING_FRAMES:
            del tracks[track_id]

    for detection in detections:
        if not detection["inside"] and detection["center"][1] < zone_top:
            recent_above.append(
                {
                    "label": detection["label"],
                    "center": detection["center"],
                    "age": 0,
                }
            )

    return drop_ready


def draw_litter(frame, detections, drop_zone):
    if drop_zone is None:
        cv2.putText(
            frame,
            "Aim camera at the recycling bin",
            (20, 190),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 255),
            2,
        )
    else:
        x1_zone, y1_zone, x2_zone, y2_zone = drop_zone
        overlay = frame.copy()
        cv2.rectangle(
            overlay, (x1_zone, y1_zone), (x2_zone, y2_zone), (255, 180, 0), -1
        )
        cv2.addWeighted(overlay, 0.18, frame, 0.82, 0, frame)
        for x in range(x1_zone, x2_zone, 24):
            cv2.line(
                frame,
                (x, y1_zone),
                (min(x + 12, x2_zone), y1_zone),
                (255, 180, 0),
                2,
            )
            cv2.line(
                frame,
                (x, y2_zone),
                (min(x + 12, x2_zone), y2_zone),
                (255, 180, 0),
                2,
            )
        for y in range(y1_zone, y2_zone, 24):
            cv2.line(
                frame,
                (x1_zone, y),
                (x1_zone, min(y + 12, y2_zone)),
                (255, 180, 0),
                2,
            )
            cv2.line(
                frame,
                (x2_zone, y),
                (x2_zone, min(y + 12, y2_zone)),
                (255, 180, 0),
                2,
            )
        cv2.putText(
            frame,
            "BIN OPENING - DROP BOTTLE HERE",
            (x1_zone, max(y1_zone - 10, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 180, 0),
            2,
        )
    for detection in detections:
        x1, y1, x2, y2 = detection["box"]
        label = detection["label"]
        inside = detection["inside"]
        if detection["kind"] == "bin":
            color = (255, 0, 255)
        else:
            color = (0, 255, 0) if inside else (0, 165, 255)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            frame,
            label,
            (x1, max(y1 - 8, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2,
        )


def load_litter_models(model_choice):
    if model_choice == MODEL_FINE_TUNED:
        if not FINE_TUNED_BIN_MODEL_PATH.is_file():
            raise FileNotFoundError(
                "Fine-tuned bin YOLO weights were not found at "
                f"{FINE_TUNED_BIN_MODEL_PATH}. Train/export the bin-only model first."
            )
        print(
            f"Loading fine-tuned bin model: {FINE_TUNED_BIN_MODEL_PATH}",
            flush=True,
        )
        bin_model = YOLO(str(FINE_TUNED_BIN_MODEL_PATH))
        model_names = getattr(bin_model, "names", None)
        if isinstance(model_names, dict):
            trained_classes = set(model_names.values())
        elif isinstance(model_names, (list, tuple)):
            trained_classes = set(model_names)
        else:
            raise RuntimeError(
                "Could not read the class names stored in the fine-tuned bin checkpoint."
            )
        if "bin" not in trained_classes:
            raise RuntimeError(
                "The bin-only checkpoint must contain a class named 'bin'. It contains: "
                f"{', '.join(sorted(trained_classes))}. Retrain from the Roboflow "
                "bin dataset and export the bin-only best.pt."
            )
        print("Loading YOLO-World recyclable-item model...", flush=True)
        item_model = YOLO("yolov8s-world.pt")
        item_model.set_classes(sorted(TARGET_CLASSES))
        print(
            "Using pretrained YOLO-World for recyclable items and fine-tuned "
            f"YOLOv8 for bins: {FINE_TUNED_BIN_MODEL_PATH}"
        )
        return item_model, bin_model

    if model_choice == MODEL_OLD:
        print("Loading pretrained YOLO-World model...", flush=True)
        model = YOLO("yolov8s-world.pt")
        model.set_classes(list(YOLO_CLASSES))
        print("Using pretrained YOLO-World")
        return model, None

    raise ValueError(f"Unknown detector selection: {model_choice}")


def run_session(
    model_choice=MODEL_OLD,
    *,
    zone_mode=ZONE_LOCK,
    no_confirm=False,
    no_face=False,
    video_path=None,
    record_directory=None,
    max_storage_bytes=int(DEFAULT_MAX_STORAGE_GB * 1024**3),
    blur=True,
    sensor=None,
):
    if zone_mode not in {ZONE_LOCK, ZONE_DETECT, ZONE_FIXED}:
        raise ValueError(f"Unknown zone mode: {zone_mode}")
    known_embeddings = [] if no_face else load_embeddings()
    if not no_face and not known_embeddings:
        raise RuntimeError(
            "No users registered. Register someone in the startup window or use "
            "python face.py --add-user Name image.jpg"
        )

    litter_model, bin_model = load_litter_models(model_choice)
    bin_detector = bin_model if bin_model is not None else litter_model
    camera = cv2.VideoCapture(str(video_path)) if video_path else cv2.VideoCapture(0)
    if not camera.isOpened():
        camera.release()
        raise RuntimeError(
            f"Could not open video source: {video_path}" if video_path
            else "Could not open the webcam"
        )

    recorder = None
    sensor_started = False
    detected_drop_times = []
    true_drop_times = []
    session_started = time.monotonic()
    frame_count = 0
    calibrated_box = None
    drop_zone = None

    try:
        if sensor is not None:
            sensor.start()
            sensor_started = True

        success, first_frame = camera.read()
        if not success:
            if video_path:
                return None
            raise RuntimeError("Could not read the first frame from the video source")
        fps = camera.get(cv2.CAP_PROP_FPS)
        if not fps or fps <= 0:
            fps = 30.0
        if record_directory is not None:
            checkpoint_paths = [
                str(FINE_TUNED_BIN_MODEL_PATH)
                if model_choice == MODEL_FINE_TUNED
                else "yolov8s-world.pt"
            ]
            recorder = SessionRecorder(
                record_directory,
                (first_frame.shape[1], first_frame.shape[0]),
                fps,
                {
                    "zone_mode": zone_mode,
                    "calibrated_bin_box": None,
                    "model_checkpoint": checkpoint_paths[0],
                    "bin_confidence": BIN_CONFIDENCE,
                    "sensor_connected": sensor is not None,
                },
                max_storage_bytes=max_storage_bytes,
                blur=blur,
                clip_seconds_before=CLIP_SECONDS_BEFORE,
                clip_seconds_after=CLIP_SECONDS_AFTER,
                segment_minutes=SEGMENT_MINUTES,
            )
            frame_info = recorder.add_frame(first_frame)
        else:
            frame_info = {"global_frame": frame_count, "time_sec": 0.0}
        frame_count += 1

        def read_next_frame():
            nonlocal frame_count
            success, next_frame = camera.read()
            if not success:
                if video_path:
                    return None
                raise RuntimeError("Could not read a frame from the webcam")
            if recorder is not None:
                info = recorder.add_frame(next_frame)
            else:
                info = {
                    "global_frame": frame_count,
                    "time_sec": max(0.0, time.monotonic() - session_started),
                }
            frame_count += 1
            return next_frame, info

        def record_event(event_type, frame_info, zone, user, labels=()):
            if recorder is not None:
                recorder.record_event(
                    event_type,
                    time_sec=frame_info["time_sec"],
                    global_frame=frame_info["global_frame"],
                    zone=zone,
                    zone_mode=zone_mode,
                    user=user,
                    item_labels=labels,
                )

        frame = first_frame
        target_bin = None
        drift_state = {"bad_checks": 0, "drifted": False}
        next_drift_check = time.monotonic() + DRIFT_CHECK_SECONDS
        if zone_mode == ZONE_FIXED:
            try:
                target_bin = select_target_bin(
                    camera, frame_provider=read_next_frame
                )
            except RuntimeError as error:
                if video_path and "Could not read a frame while selecting" in str(error):
                    return recorder.directory if recorder is not None else None
                raise
            calibrated_box = target_bin["box"]
            drop_zone = get_drop_zone(frame, target_bin)
            record_event("calibrated", frame_info, drop_zone, None)
        elif zone_mode == ZONE_LOCK:
            try:
                calibrated_box, reason, calibration_frame, calibration_info = calibrate_bin(
                    camera,
                    bin_detector,
                    frame_provider=read_next_frame,
                    initial_frame=frame,
                    frame_count=CALIBRATION_FRAMES,
                )
            except EOFError:
                return recorder.directory if recorder is not None else None
            if calibrated_box is None:
                print(
                    f"Automatic bin calibration failed ({reason}); "
                    "select the bin manually."
                )
                try:
                    target_bin = select_target_bin(
                        camera, frame_provider=read_next_frame
                    )
                except RuntimeError as error:
                    if video_path and "Could not read a frame while selecting" in str(error):
                        return recorder.directory if recorder is not None else None
                    raise
            else:
                target_bin = confirm_bin(
                    camera,
                    calibration_frame,
                    calibrated_box,
                    no_confirm=no_confirm,
                    frame_provider=read_next_frame,
                )
            calibrated_box = target_bin["box"]
            drop_zone = get_drop_zone(calibration_frame, target_bin)
            frame_info = calibration_info or frame_info
            record_event("calibrated", frame_info, drop_zone, None)
            frame = calibration_frame
        if recorder is not None:
            session_metadata_path = recorder.directory / "session.json"
            session_metadata = json.loads(
                session_metadata_path.read_text(encoding="utf-8")
            )
            session_metadata["calibrated_bin_box"] = calibrated_box
            session_metadata["drop_zone"] = drop_zone
            session_metadata["zone_mode"] = zone_mode
            session_metadata_path.write_text(
                json.dumps(session_metadata, indent=2), encoding="utf-8"
            )

        status = "Face recognition disabled" if no_face else "Loading face model..."
        distance = None
        facial_area = None
        detections = []
        drop_tracks = {}
        recent_above_detections = []
        last_award_time = -DROP_COOLDOWN_SECONDS
        banner = (
            "Self-check: no sensor - review clips after the session"
            if sensor is None
            else ""
        )
        banner_until = float("inf") if sensor is None else 0.0
        user_points = 0
        active_user = None
        zone_activity_started = None
        zone_activity_recorded = False

        while frame is not None:
            loop_time = time.monotonic()
            if not no_face and frame_count % 10 == 0:
                status, distance, facial_area = recognize_frame(
                    frame, known_embeddings
                )
                if status.startswith("Identified:"):
                    active_user = status.partition(":")[2].strip()

            inside_items = []
            if frame_count % YOLO_FRAME_INTERVAL == 0:
                if zone_mode == ZONE_DETECT:
                    bin_detections = detect_bins(
                        bin_detector, frame, BIN_CONFIDENCE
                    )
                    bin_detection = (
                        max(bin_detections, key=lambda detection: detection["confidence"])
                        if bin_detections
                        else None
                    )
                    if bin_detection is not None:
                        calibrated_box = bin_detection["box"]
                        box = bin_detection["box"]
                        bin_height = box[3] - box[1]
                    else:
                        bin_height = None
                    drop_zone = get_drop_zone(frame, bin_detection)
                    item_detections = detect_items(
                        litter_model, frame, drop_zone
                    )
                    detections = item_detections + bin_detections
                elif drop_zone is not None:
                    bin_height = target_bin["box"][3] - target_bin["box"][1]
                    item_detections = detect_items(
                        litter_model, frame, drop_zone
                    )
                    detections = item_detections + [dict(target_bin)]
                else:
                    bin_height = None
                    item_detections = detect_items(
                        litter_model, frame, drop_zone
                    )
                    detections = item_detections

                mark_detections_in_drop_zone(detections, drop_zone)
                inside_items = [
                    detection
                    for detection in item_detections
                    if detection["inside"]
                ]
                if zone_mode == ZONE_LOCK and not drift_state["drifted"]:
                    if inside_items:
                        if zone_activity_started is None:
                            zone_activity_started = loop_time
                        elif (
                            not zone_activity_recorded
                            and loop_time - zone_activity_started
                            >= ZONE_ACTIVITY_TIMEOUT
                        ):
                            record_event(
                                "zone_activity",
                                frame_info,
                                drop_zone,
                                active_user,
                                [detection["label"] for detection in inside_items],
                            )
                            zone_activity_recorded = True
                    else:
                        zone_activity_started = None
                        zone_activity_recorded = False

                if drop_zone is None or drift_state["drifted"]:
                    drop_tracks.clear()
                    recent_above_detections.clear()
                    drop_ready = False
                else:
                    drop_ready = update_drop_tracks(
                        drop_tracks,
                        item_detections,
                        drop_zone,
                        bin_height,
                        recent_above_detections,
                    )
                if drop_ready:
                    detected_drop_times.append(frame_info["time_sec"])
                    record_event(
                        "drop_detected",
                        frame_info,
                        drop_zone,
                        active_user,
                        [
                            detection["label"]
                            for detection in item_detections
                            if detection["inside"]
                        ],
                    )
                    if (
                        not drift_state["drifted"]
                        and active_user
                        and loop_time - last_award_time >= DROP_COOLDOWN_SECONDS
                    ):
                        user_points = award_points(active_user)
                        banner = "RECYCLABLE DROPPED: +10 Points Awarded!"
                        banner_until = loop_time + 3.0
                        last_award_time = loop_time

            if (
                zone_mode == ZONE_LOCK
                and not drift_state["drifted"]
                and loop_time >= next_drift_check
                and not any(track["drop_candidate"] for track in drop_tracks.values())
                and not inside_items
            ):
                drift_detections = detect_bins(
                    bin_detector, frame, BIN_CONFIDENCE
                )
                best_drift_box = (
                    max(
                        drift_detections,
                        key=lambda detection: detection["confidence"],
                    )["box"]
                    if drift_detections
                    else None
                )
                drift_iou = (
                    box_iou(calibrated_box, best_drift_box)
                    if best_drift_box is not None
                    else None
                )
                previous_drift = drift_state["drifted"]
                drift_state = update_drift(drift_state, drift_iou)
                if drift_state["drifted"] and not previous_drift:
                    banner = "Bin moved - press C to recalibrate"
                    banner_until = float("inf")
                    record_event("drift_detected", frame_info, drop_zone, active_user)
                    drop_tracks.clear()
                    recent_above_detections.clear()
                next_drift_check = loop_time + DRIFT_CHECK_SECONDS

            if sensor is not None:
                for sensor_event in sensor.drain_events():
                    if sensor_event.active:
                        sensor_origin = (
                            recorder.started_at
                            if recorder is not None
                            else session_started
                        )
                        event_time = max(
                            0.0, sensor_event.received_at - sensor_origin
                        )
                        true_drop_times.append(event_time)
                        record_event(
                            "sensor_drop",
                            {
                                "time_sec": event_time,
                                "global_frame": frame_info["global_frame"],
                            },
                            drop_zone,
                            active_user,
                        )

            if active_user and not no_face:
                user_points = get_user_points(active_user)

            identified = status.startswith("Identified:")
            display_frame = frame.copy()
            draw_face_box(display_frame, facial_area, identified)
            draw_litter(display_frame, detections, drop_zone)
            display_status = status
            if distance is not None:
                display_status = f"{status} ({distance:.3f})"
            cv2.putText(
                display_frame,
                f"Points: {user_points}",
                (20, 105),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0) if identified else (255, 255, 255),
                2,
            )
            cv2.putText(
                display_frame,
                display_status,
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0) if identified else (0, 0, 255),
                2,
            )
            if sensor is not None:
                from session_metrics import match_events

                matches, unmatched_detected, unmatched_true = match_events(
                    detected_drop_times, true_drop_times, MATCH_TOLERANCE_SECONDS
                )
                true_positive = len(matches)
                false_positive = len(unmatched_detected)
                false_negative = len(unmatched_true)
                self_check = (
                    f"TP {true_positive} | FP {false_positive} | FN {false_negative}"
                )
                cv2.putText(
                    display_frame,
                    self_check,
                    (20, 135),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2,
                )
            if time.monotonic() < banner_until or drift_state["drifted"]:
                cv2.rectangle(
                    display_frame,
                    (10, 145),
                    (frame.shape[1] - 10, 190),
                    (0, 90, 160),
                    -1,
                )
                cv2.putText(
                    display_frame,
                    banner,
                    (20, 175),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (255, 255, 255),
                    2,
                )
            cv2.putText(
                display_frame,
                "Press Q or Esc to quit",
                (20, 75),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                1,
            )
            cv2.imshow("Face recognition", display_frame)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key in (ord("c"), ord("C")) and zone_mode == ZONE_LOCK:
                try:
                    calibrated_box, reason, calibration_frame, calibration_info = calibrate_bin(
                        camera,
                        bin_detector,
                        frame_provider=read_next_frame,
                        initial_frame=frame,
                        frame_count=CALIBRATION_FRAMES,
                    )
                except EOFError:
                    break
                if calibrated_box is None:
                    print(
                        f"Recalibration failed ({reason}); select the bin manually."
                    )
                    target_bin = select_target_bin(
                        camera, frame_provider=read_next_frame
                    )
                else:
                    target_bin = confirm_bin(
                        camera,
                        calibration_frame,
                        calibrated_box,
                        no_confirm=no_confirm,
                        frame_provider=read_next_frame,
                    )
                calibrated_box = target_bin["box"]
                drop_zone = get_drop_zone(calibration_frame, target_bin)
                drift_state = {"bad_checks": 0, "drifted": False}
                drop_tracks.clear()
                recent_above_detections.clear()
                zone_activity_started = None
                zone_activity_recorded = False
                frame = calibration_frame
                frame_info = calibration_info or frame_info
                next_drift_check = time.monotonic() + DRIFT_CHECK_SECONDS
                if recorder is not None:
                    metadata_path = recorder.directory / "session.json"
                    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
                    metadata["calibrated_bin_box"] = calibrated_box
                    metadata["drop_zone"] = drop_zone
                    metadata_path.write_text(
                        json.dumps(metadata, indent=2), encoding="utf-8"
                    )
                record_event("recalibrated", frame_info, drop_zone, active_user)
                banner = "Bin recalibrated"
                banner_until = time.monotonic() + 3.0
                continue
            next_frame = read_next_frame()
            if next_frame is None:
                break
            frame, frame_info = next_frame
    finally:
        camera.release()
        cv2.destroyAllWindows()
        if sensor_started:
            sensor.close()
        if recorder is not None:
            metadata_path = recorder.directory / "session.json"
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            metadata["calibrated_bin_box"] = calibrated_box
            metadata["drop_zone"] = drop_zone
            metadata_path.write_text(
                json.dumps(metadata, indent=2), encoding="utf-8"
            )
            recorder.close()

    if recorder is not None and sensor is not None:
        from session_metrics import match_events

        matches, unmatched_detected, unmatched_true = match_events(
            detected_drop_times, true_drop_times, MATCH_TOLERANCE_SECONDS
        )
        true_positive = len(matches)
        false_positive = len(unmatched_detected)
        false_negative = len(unmatched_true)
        delays = [
            detected_drop_times[detected_index] - true_drop_times[true_index]
            for detected_index, true_index, _difference in matches
        ]
        precision = (
            true_positive / (true_positive + false_positive)
            if true_positive + false_positive
            else 0.0
        )
        recall = (
            true_positive / (true_positive + false_negative)
            if true_positive + false_negative
            else 0.0
        )
        f1 = (
            2 * precision * recall / (precision + recall)
            if precision + recall
            else 0.0
        )
        with (recorder.directory / "live_report.csv").open(
            "w", newline="", encoding="utf-8"
        ) as report_file:
            report_writer = csv.writer(report_file)
            report_writer.writerow(
                [
                    "true_positive",
                    "false_positive",
                    "false_negative",
                    "precision",
                    "recall",
                    "f1",
                    "mean_detection_delay",
                ]
            )
            report_writer.writerow(
                [
                    true_positive,
                    false_positive,
                    false_negative,
                    f"{precision:.6f}",
                    f"{recall:.6f}",
                    f"{f1:.6f}",
                    f"{float(np.mean(delays)):.6f}" if delays else "",
                ]
            )
    return recorder.directory if recorder is not None else None


def run_webcam(model_choice=MODEL_OLD, **kwargs):
    return run_session(model_choice, **kwargs)


def run_model_comparison_webcam():
    if not FINE_TUNED_BIN_MODEL_PATH.is_file():
        raise FileNotFoundError(
            "Fine-tuned bin YOLO weights were not found at "
            f"{FINE_TUNED_BIN_MODEL_PATH}. Train/export the bin-only model first."
        )

    known_embeddings = load_embeddings()
    if not known_embeddings:
        raise RuntimeError(
            "No users registered. Register someone in the startup window or use "
            "python face.py --add-user Name image.jpg"
        )

    old_item_model, old_bin_model = load_litter_models(MODEL_OLD)
    new_item_model, new_bin_model = load_litter_models(MODEL_FINE_TUNED)
    camera = cv2.VideoCapture(0)
    if not camera.isOpened():
        raise RuntimeError("Could not open the webcam")

    fine_tuned_name = "YOLO-World items + YOLOv8 bin"
    latencies = {"YOLO-World (old)": [], fine_tuned_name: []}
    bin_frames = {"YOLO-World (old)": 0, fine_tuned_name: 0}
    frame_count = 0
    warmed_up = False
    face_status = "Loading face model..."
    face_distance = None
    facial_area = None
    try:
        target_bin = select_target_bin(camera)

        while True:
            success, frame = camera.read()
            if not success:
                raise RuntimeError("Could not read a frame from the webcam")
            if not warmed_up:
                old_item_model(frame, verbose=False, conf=DETECTION_CONFIDENCE)
                new_item_model(frame, verbose=False, conf=DETECTION_CONFIDENCE)
                new_bin_model(frame, verbose=False, conf=DETECTION_CONFIDENCE)
                warmed_up = True
            frame_count += 1
            if frame_count % 10 == 0:
                face_status, face_distance, facial_area = recognize_frame(
                    frame, known_embeddings
                )

            model_pairs = [
                ("YOLO-World (old)", old_item_model, old_bin_model),
                (
                    "YOLO-World items + YOLOv8 bin",
                    new_item_model,
                    new_bin_model,
                ),
            ]
            if frame_count % 2 == 0:
                model_pairs.reverse()

            results = {}
            for model_name, item_model, bin_model in model_pairs:
                started = time.perf_counter()
                model_detections = detect_litter(
                    item_model, frame, None, bin_model=bin_model
                )
                elapsed_ms = (time.perf_counter() - started) * 1000
                bin_detections = [
                    detection for detection in model_detections
                    if detection["kind"] == "bin"
                ]
                latencies[model_name].append(elapsed_ms)
                if bin_detections:
                    bin_frames[model_name] += 1
                detections = keep_selected_bin(model_detections, target_bin)
                results[model_name] = (detections, elapsed_ms, bin_detections)

            panels = []
            for model_name in (
                "YOLO-World (old)",
                "YOLO-World items + YOLOv8 bin",
            ):
                detections, elapsed_ms, bin_detections = results[model_name]
                panel = frame.copy()
                panel_detections = [dict(detection) for detection in detections]
                drop_zone = get_drop_zone(frame, target_bin)
                mark_detections_in_drop_zone(panel_detections, drop_zone)
                draw_litter(panel, panel_detections, drop_zone)
                identified = face_status.startswith("Identified:")
                draw_face_box(panel, facial_area, identified)
                bin_status = f"Selected bin | model bin boxes: {len(bin_detections)}"
                cv2.rectangle(panel, (0, 0), (panel.shape[1], 100), (30, 30, 30), -1)
                cv2.putText(
                    panel,
                    f"{model_name} | inference: {elapsed_ms:.1f} ms",
                    (12, 27),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.62,
                    (255, 255, 255),
                    2,
                )
                cv2.putText(
                    panel,
                    bin_status,
                    (12, 54),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.62,
                    (0, 255, 0),
                    2,
                )
                display_face_status = f"Face: {face_status}"
                if face_distance is not None:
                    display_face_status += f" ({face_distance:.3f})"
                cv2.putText(
                    panel,
                    display_face_status,
                    (12, 88),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.62,
                    (0, 255, 0) if identified else (0, 0, 255),
                    2,
                )
                panels.append(panel)

            comparison = cv2.hconcat(panels)
            cv2.putText(
                comparison,
                "Same camera frame | Q or Esc to finish and show timing summary",
                (12, comparison.shape[0] - 14),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.58,
                (255, 255, 255),
                2,
            )
            cv2.imshow("Live YOLO detector comparison", comparison)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
    finally:
        camera.release()
        cv2.destroyAllWindows()

    print("\nLive comparison summary")
    print(f"Frames compared: {frame_count}")
    for model_name in (
        "YOLO-World (old)",
        fine_tuned_name,
    ):
        samples = latencies[model_name]
        if not samples:
            continue
        detection_rate = 100 * bin_frames[model_name] / len(samples)
        print(
            f"{model_name}: mean {np.mean(samples):.1f} ms, "
            f"median {np.median(samples):.1f} ms, "
            f"p95 {np.percentile(samples, 95):.1f} ms; "
            f"bin detected in {bin_frames[model_name]}/{len(samples)} frames "
            f"({detection_rate:.1f}%)"
        )
    print(
        "Bin detection rate is only the fraction of frames with a detection, "
        "not accuracy; check the video for false positives and missed bins."
    )


def show_registration_window():
    import tkinter as tk
    from tkinter import filedialog, messagebox, ttk

    root = tk.Tk()
    root.title("Register a person")
    root.resizable(False, False)

    name = tk.StringVar()
    image_path = tk.StringVar()
    start_requested = False
    selected_model = None
    model_choice = tk.StringVar(value=MODEL_OLD)

    content = ttk.Frame(root, padding=16)
    content.grid(sticky="nsew")
    content.columnconfigure(1, weight=1)

    ttk.Label(content, text="Name").grid(row=0, column=0, sticky="w", padx=(0, 8), pady=6)
    name_entry = ttk.Entry(content, textvariable=name, width=38)
    name_entry.grid(row=0, column=1, columnspan=2, sticky="ew", pady=6)

    ttk.Label(content, text="Photo").grid(row=1, column=0, sticky="w", padx=(0, 8), pady=6)
    ttk.Entry(content, textvariable=image_path, width=38).grid(
        row=1, column=1, sticky="ew", pady=6
    )

    def choose_photo():
        selected_path = filedialog.askopenfilename(
            parent=root,
            title="Choose a face photo",
            filetypes=[
                ("Image files", "*.jpg *.jpeg *.png *.bmp *.webp"),
                ("All files", "*.*"),
            ],
        )
        if selected_path:
            image_path.set(selected_path)

    ttk.Button(content, text="Browse...", command=choose_photo).grid(
        row=1, column=2, padx=(8, 0), pady=6
    )

    status = ttk.Label(content, text="Choose a clear photo with one visible face.")
    status.grid(row=2, column=0, columnspan=3, sticky="w", pady=(8, 12))

    ttk.Label(content, text="Object detector").grid(
        row=3, column=0, columnspan=3, sticky="w", pady=(4, 2)
    )
    detector_options = (
        ("Pretrained YOLO-World (old)", MODEL_OLD),
        ("YOLO-World items + fine-tuned YOLOv8 bin", MODEL_FINE_TUNED),
        ("Compare both side by side", MODEL_COMPARE),
    )
    for row, (label, value) in enumerate(detector_options, start=4):
        ttk.Radiobutton(
            content, text=label, variable=model_choice, value=value
        ).grid(row=row, column=0, columnspan=3, sticky="w")
    ttk.Label(
        content,
        text=(
            "Pretrained mode uses YOLO-World for recyclables and bins. "
            "Fine-tuned mode calibrates a bin at startup; comparison asks you "
            "to select one. The default locked zone is calibrated once. "
            f"Fine-tuned weights: {FINE_TUNED_BIN_MODEL_PATH}"
        ),
        wraplength=430,
    ).grid(row=7, column=0, columnspan=3, sticky="w", pady=(4, 8))

    def register_person():
        clean_name = name.get().strip()
        selected_path = image_path.get().strip()
        if not clean_name:
            messagebox.showerror("Name required", "Enter the person's name.", parent=root)
            name_entry.focus_set()
            return
        if not selected_path:
            messagebox.showerror("Photo required", "Choose a photo first.", parent=root)
            return

        register_button.config(state="disabled")
        status.config(text="Processing photo...")
        root.update_idletasks()
        try:
            add_user(clean_name, selected_path)
        except (FileNotFoundError, RuntimeError, ValueError) as error:
            status.config(text="Registration failed.")
            messagebox.showerror("Could not register person", str(error), parent=root)
        else:
            status.config(text=f"{clean_name} is registered. You can add another person.")
            messagebox.showinfo(
                "Registration complete",
                f"{clean_name} has been registered.",
                parent=root,
            )
        finally:
            register_button.config(state="normal")

    register_button = ttk.Button(
        content, text="Register person", command=register_person
    )
    register_button.grid(row=8, column=0, sticky="ew", padx=(0, 6))

    def continue_to_recognition():
        nonlocal selected_model, start_requested
        if (
            model_choice.get() in {MODEL_FINE_TUNED, MODEL_COMPARE}
            and not FINE_TUNED_BIN_MODEL_PATH.is_file()
        ):
            messagebox.showerror(
                "Fine-tuned model not found",
                "Train/export the fine-tuned YOLO model and place it at:\n"
                f"{FINE_TUNED_BIN_MODEL_PATH}",
                parent=root,
            )
            return
        selected_model = model_choice.get()
        start_requested = True
        root.destroy()

    ttk.Button(
        content, text="Start webcam", command=continue_to_recognition
    ).grid(row=8, column=1, columnspan=2, sticky="ew", padx=(6, 0))

    name_entry.focus_set()
    root.mainloop()
    return selected_model if start_requested else None


def main():
    parser = argparse.ArgumentParser(description="Recognize registered users by webcam")
    parser.add_argument("--add-user", nargs=2, metavar=("NAME", "IMAGE"))
    parser.add_argument(
        "--zone-mode",
        choices=(ZONE_LOCK, ZONE_DETECT, ZONE_FIXED),
        default=ZONE_LOCK,
    )
    parser.add_argument("--record", nargs="?", const="sessions", metavar="DIR")
    parser.add_argument(
        "--max-storage-gb", type=float, default=DEFAULT_MAX_STORAGE_GB
    )
    parser.add_argument("--no-blur", action="store_true")
    parser.add_argument("--sensor", help="mock or serial:PORT")
    parser.add_argument("--video", help="Read a video file instead of the webcam")
    parser.add_argument("--no-face", action="store_true")
    parser.add_argument("--no-confirm", action="store_true")
    args = parser.parse_args()

    try:
        if args.add_user:
            add_user(*args.add_user)
        else:
            if args.max_storage_gb <= 0:
                parser.error("--max-storage-gb must be greater than zero")
            sensor = None
            if args.sensor == "mock":
                sensor = MockSensorSource()
            elif args.sensor:
                if not args.sensor.startswith("serial:"):
                    parser.error("--sensor must be 'mock' or 'serial:PORT'")
                port = args.sensor.partition(":")[2].strip()
                if not port:
                    parser.error("Serial sensor port cannot be empty")
                sensor = SerialSensorSource(port)

            if args.video:
                model_choice = MODEL_FINE_TUNED
            elif args.no_face:
                model_choice = MODEL_OLD
            else:
                model_choice = show_registration_window()

            if model_choice == MODEL_COMPARE:
                if (
                    args.record
                    or args.sensor
                    or args.no_face
                    or args.no_blur
                    or args.zone_mode != ZONE_LOCK
                    or args.no_confirm
                ):
                    parser.error(
                        "Recording, sensors, face disabling, and zone-mode "
                        "options are not supported in comparison mode"
                    )
                run_model_comparison_webcam()
            elif model_choice in {MODEL_OLD, MODEL_FINE_TUNED}:
                run_session(
                    model_choice,
                    zone_mode=args.zone_mode,
                    no_confirm=args.no_confirm,
                    no_face=args.no_face,
                    video_path=args.video,
                    record_directory=args.record,
                    max_storage_bytes=int(args.max_storage_gb * 1024**3),
                    blur=not args.no_blur,
                    sensor=sensor,
                )
    except (FileNotFoundError, RuntimeError, ValueError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
