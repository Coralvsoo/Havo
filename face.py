import argparse
import json
import sqlite3
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from ultralytics import YOLO

PROJECT_DIR = Path(__file__).resolve().parent
LOCAL_DEEPFACE_SOURCE = PROJECT_DIR / "deepface"
if (LOCAL_DEEPFACE_SOURCE / "deepface" / "__init__.py").is_file():
    sys.path.insert(0, str(LOCAL_DEEPFACE_SOURCE))

from deepface import DeepFace


DATABASE_PATH = PROJECT_DIR / "smartclean_app.db"
MODEL_NAME = "Facenet"
MATCH_THRESHOLD = 0.40
POINTS_PER_DROP = 10
DROP_COOLDOWN_SECONDS = 3.0
YOLO_FRAME_INTERVAL = 4
TARGET_CLASSES = {"bottle", "cup", "can"}
BIN_CLASSES = {"trash can", "recycling bin"}
YOLO_CLASSES = ("bottle", "cup", "can", "trash can", "recycling bin")
DROP_ZONE = (200, 300, 440, 480)
TRACK_MATCH_DISTANCE = 100
DROP_MIN_DOWNWARD_PIXELS = 15
DROP_SETTLE_DISTANCE = 12
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


def add_user(name, image_path):
    clean_name = name.strip()
    if not clean_name:
        raise ValueError("User name cannot be empty")

    source = Path(image_path).resolve()
    if not source.is_file():
        raise FileNotFoundError(f"Image not found: {source}")

    try:
        embedding = extract_embedding(str(source))
        vector_json = json.dumps(embedding.tolist())
        with connect_database() as connection:
            connection.execute(
                """
                INSERT INTO face_embeddings (user_name, vector_json)
                VALUES (?, ?)
                ON CONFLICT(user_name) DO UPDATE SET vector_json = excluded.vector_json
                """,
                (clean_name, vector_json),
            )
            connection.commit()
        print(f"Registered {clean_name} in {DATABASE_PATH}")
    except (sqlite3.Error, OSError, ValueError) as error:
        raise RuntimeError(f"Could not register {clean_name}: {error}") from error


def load_embeddings():
    try:
        with connect_database() as connection:
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
        with connect_database() as connection:
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
        with connect_database() as connection:
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


def get_drop_zone(frame):
    height, width = frame.shape[:2]
    return (
        min(DROP_ZONE[0], width - 1),
        min(DROP_ZONE[1], height - 1),
        min(DROP_ZONE[2], width),
        min(DROP_ZONE[3], height),
    )


def detect_litter(model, frame, drop_zone):
    detections = []
    x1_zone, y1_zone, x2_zone, y2_zone = drop_zone
    result = model(frame, verbose=False)[0]

    for box in result.boxes:
        class_id = int(box.cls[0].item())
        label = result.names[class_id]
        if label not in TARGET_CLASSES and label not in BIN_CLASSES:
            continue
        if float(box.conf[0].item()) < 0.35:
            continue

        x1, y1, x2, y2 = [int(value) for value in box.xyxy[0].tolist()]
        center_x = (x1 + x2) // 2
        center_y = (y1 + y2) // 2
        inside = x1_zone <= center_x <= x2_zone and y1_zone <= center_y <= y2_zone
        detections.append(
            {
                "box": (x1, y1, x2, y2),
                "label": label,
                "kind": "bin" if label in BIN_CLASSES else "recyclable",
                "center": (center_x, center_y),
                "inside": inside,
            }
        )

    return detections


def update_drop_tracks(tracks, detections, drop_zone):
    matched_track_ids = set()
    drop_ready = False
    zone_top = drop_zone[1]

    for detection in detections:
        center_x, center_y = detection["center"]
        best_track = None
        best_distance = TRACK_MATCH_DISTANCE
        for track_id, track in tracks.items():
            if track_id in matched_track_ids or track["label"] != detection["label"]:
                continue
            distance = np.hypot(
                center_x - track["center"][0], center_y - track["center"][1]
            )
            if distance < best_distance:
                best_track = track_id
                best_distance = distance

        if best_track is None:
            best_track = max(tracks.keys(), default=-1) + 1
            tracks[best_track] = {
                "label": detection["label"],
                "center": detection["center"],
                "was_inside": False,
                "drop_candidate": False,
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
                and center_y - previous_y >= DROP_MIN_DOWNWARD_PIXELS
            ):
                track["drop_candidate"] = True
                track["settled_frames"] = 0
            elif track["drop_candidate"] and not track["drop_confirmed"]:
                movement = np.hypot(center_x - previous_x, center_y - previous_y)
                if movement <= DROP_SETTLE_DISTANCE:
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
        if track["missing_frames"] > 3:
            del tracks[track_id]

    return drop_ready


def draw_litter(frame, detections, drop_zone):
    x1_zone, y1_zone, x2_zone, y2_zone = drop_zone
    overlay = frame.copy()
    cv2.rectangle(overlay, (x1_zone, y1_zone), (x2_zone, y2_zone), (255, 180, 0), -1)
    cv2.addWeighted(overlay, 0.18, frame, 0.82, 0, frame)
    for x in range(x1_zone, x2_zone, 24):
        cv2.line(frame, (x, y1_zone), (min(x + 12, x2_zone), y1_zone), (255, 180, 0), 2)
        cv2.line(frame, (x, y2_zone), (min(x + 12, x2_zone), y2_zone), (255, 180, 0), 2)
    for y in range(y1_zone, y2_zone, 24):
        cv2.line(frame, (x1_zone, y), (x1_zone, min(y + 12, y2_zone)), (255, 180, 0), 2)
        cv2.line(frame, (x2_zone, y), (x2_zone, min(y + 12, y2_zone)), (255, 180, 0), 2)
    cv2.putText(
        frame,
        "RECYCLING DROP ZONE",
        (x1_zone, y1_zone - 10),
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


def run_webcam():
    known_embeddings = load_embeddings()
    if not known_embeddings:
        raise RuntimeError(
            "No users registered. Register someone in the startup window or use "
            "python face.py --add-user Name image.jpg"
        )

    camera = cv2.VideoCapture(0)
    if not camera.isOpened():
        raise RuntimeError("Could not open the webcam")

    litter_model = YOLO("yolov8s-world.pt")
    litter_model.set_classes(list(YOLO_CLASSES))
    status = "Loading face model..."
    distance = None
    facial_area = None
    frame_count = 0
    detections = []
    drop_zone = get_drop_zone(np.zeros((480, 640, 3), dtype=np.uint8))
    drop_tracks = {}
    last_award_time = -DROP_COOLDOWN_SECONDS
    banner = ""
    banner_until = 0.0
    user_points = 0
    active_user = None

    try:
        while True:
            success, frame = camera.read()
            if not success:
                raise RuntimeError("Could not read a frame from the webcam")

            frame_count += 1
            if frame_count % 10 == 0:
                status, distance, facial_area = recognize_frame(frame, known_embeddings)
                if status.startswith("Identified:"):
                    active_user = status.partition(":")[2].strip()

            if frame_count % YOLO_FRAME_INTERVAL == 0:
                drop_zone = get_drop_zone(frame)
                detections = detect_litter(litter_model, frame, drop_zone)
                recyclable_detections = [
                    detection
                    for detection in detections
                    if detection["kind"] == "recyclable"
                ]
                drop_ready = update_drop_tracks(
                    drop_tracks, recyclable_detections, drop_zone
                )
                if drop_ready:
                    now = time.monotonic()
                    if active_user and now - last_award_time >= DROP_COOLDOWN_SECONDS:
                        user_points = award_points(active_user)
                        banner = f"RECYCLABLE DROPPED: +10 Points Awarded!"
                        banner_until = now + 3.0
                        last_award_time = now

            if active_user:
                user_points = get_user_points(active_user)

            identified = status.startswith("Identified:")
            draw_face_box(frame, facial_area, identified)
            draw_litter(frame, detections, drop_zone)
            display_status = status
            if distance is not None:
                display_status = f"{status} ({distance:.3f})"
            cv2.putText(
                frame,
                f"Points: {user_points}",
                (20, 105),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0) if identified else (255, 255, 255),
                2,
            )
            cv2.putText(
                frame,
                display_status,
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0) if identified else (0, 0, 255),
                2,
            )
            if time.monotonic() < banner_until:
                cv2.rectangle(frame, (10, 115), (frame.shape[1] - 10, 160), (0, 120, 0), -1)
                cv2.putText(
                    frame,
                    banner,
                    (20, 145),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (255, 255, 255),
                    2,
                )
            cv2.putText(
                frame,
                "Press Q or Esc to quit",
                (20, 75),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                1,
            )
            cv2.imshow("Face recognition", frame)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
    finally:
        camera.release()
        cv2.destroyAllWindows()


def show_registration_window():
    root = tk.Tk()
    root.title("Register a person")
    root.resizable(False, False)

    name = tk.StringVar()
    image_path = tk.StringVar()
    start_recognition = False

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
    register_button.grid(row=3, column=0, sticky="ew", padx=(0, 6))

    def continue_to_recognition():
        nonlocal start_recognition
        start_recognition = True
        root.destroy()

    ttk.Button(
        content, text="Start webcam", command=continue_to_recognition
    ).grid(row=3, column=1, columnspan=2, sticky="ew", padx=(6, 0))

    name_entry.focus_set()
    root.mainloop()
    return start_recognition


def main():
    parser = argparse.ArgumentParser(description="Recognize registered users by webcam")
    parser.add_argument("--add-user", nargs=2, metavar=("NAME", "IMAGE"))
    args = parser.parse_args()

    try:
        if args.add_user:
            add_user(*args.add_user)
        else:
            if show_registration_window():
                run_webcam()
    except (FileNotFoundError, RuntimeError, ValueError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
