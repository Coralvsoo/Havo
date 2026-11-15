import hmac
import logging
import os
import re
import threading
import time

import cv2
import numpy as np
from fastapi import Depends, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.security import APIKeyHeader

import face


logger = logging.getLogger(__name__)
app = FastAPI(
    title="Havo Recycling and Face Recognition API",
    description=(
        "Register a face, process uploaded webcam frames for face and recycling "
        "detection, and retrieve user points."
    ),
    version="1.0.0",
)

MAX_IMAGE_BYTES = 10 * 1024 * 1024
MAX_IMAGE_SIDE = 4096
MAX_IMAGE_PIXELS = 12_000_000
MAX_ACTIVE_SESSIONS = 1000
SESSION_EXPIRY_SECONDS = 30 * 60
SESSION_ID_PATTERN = re.compile(r"^[A-Za-z0-9._:-]{1,64}$")

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)
model_lock = threading.Lock()
session_lock = threading.Lock()
detector_models = None
sessions = {}


def require_api_key(api_key: str | None = Depends(api_key_header)):
    expected_key = os.environ.get("HAVO_API_KEY")
    if not expected_key:
        raise HTTPException(
            status_code=503,
            detail="API authentication is not configured. Set HAVO_API_KEY.",
        )
    if not api_key or not hmac.compare_digest(api_key, expected_key):
        raise HTTPException(status_code=401, detail="A valid X-API-Key is required.")


def decode_image(image: UploadFile):
    contents = image.file.read(MAX_IMAGE_BYTES + 1)
    if not contents:
        raise HTTPException(status_code=400, detail="The uploaded image is empty.")
    if len(contents) > MAX_IMAGE_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"Image is too large; maximum size is {MAX_IMAGE_BYTES} bytes.",
        )

    frame = cv2.imdecode(np.frombuffer(contents, dtype=np.uint8), cv2.IMREAD_COLOR)
    if frame is None:
        raise HTTPException(
            status_code=415,
            detail="The upload is not a supported image. Use JPEG, PNG, or WebP.",
        )
    height, width = frame.shape[:2]
    if (
        width > MAX_IMAGE_SIDE
        or height > MAX_IMAGE_SIDE
        or width * height > MAX_IMAGE_PIXELS
    ):
        raise HTTPException(
            status_code=413,
            detail=(
                f"Image dimensions must be at most {MAX_IMAGE_SIDE} pixels per side "
                f"and {MAX_IMAGE_PIXELS} pixels total."
            ),
        )
    return frame


def get_detector_models():
    global detector_models
    if detector_models is None:
        detector_models = face.load_litter_models(face.MODEL_OLD)
    return detector_models


def get_session_state(session_id: str):
    if not SESSION_ID_PATTERN.fullmatch(session_id):
        raise HTTPException(
            status_code=422,
            detail="session_id may contain only letters, digits, dot, underscore, colon, or hyphen.",
        )

    now = time.monotonic()
    with session_lock:
        expired_sessions = [
            key
            for key, state in sessions.items()
            if now - state["last_used"] > SESSION_EXPIRY_SECONDS
        ]
        for key in expired_sessions:
            del sessions[key]

        if session_id not in sessions and len(sessions) >= MAX_ACTIVE_SESSIONS:
            raise HTTPException(
                status_code=503,
                detail="The API has reached its active session limit. Try again later.",
            )
        state = sessions.setdefault(
            session_id,
            {
                "tracks": {},
                "last_award_by_user": {},
                "last_used": now,
                "lock": threading.Lock(),
            },
        )
        state["last_used"] = now
        return state


@app.get("/health", tags=["service"])
def health():
    return {"status": "ok"}


@app.post(
    "/api/v1/users/register",
    tags=["users"],
    dependencies=[Depends(require_api_key)],
)
def register_user(
    user_name: str = Form(..., min_length=1, max_length=80),
    image: UploadFile = File(..., description="A clear photo showing one face."),
):
    clean_name = user_name.strip()
    if not clean_name:
        raise HTTPException(status_code=422, detail="user_name cannot be blank.")

    frame = decode_image(image)
    try:
        with model_lock:
            embedding = face.extract_embedding(frame)
            saved_name = face.save_user_embedding(clean_name, embedding)
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    except RuntimeError as error:
        logger.exception("Could not save face registration")
        raise HTTPException(status_code=500, detail="Could not save face registration.") from error
    except Exception as error:
        logger.exception("Could not extract a face from the registration image")
        raise HTTPException(
            status_code=422,
            detail="Could not extract a face from the uploaded image.",
        ) from error

    return {
        "user_name": saved_name,
        "registered": True,
        "note": "The uploaded image was not stored; only its face embedding was saved.",
    }


@app.get(
    "/api/v1/users/{user_name}/points",
    tags=["users"],
    dependencies=[Depends(require_api_key)],
)
def user_points(user_name: str):
    clean_name = user_name.strip()
    if not clean_name:
        raise HTTPException(status_code=422, detail="user_name cannot be blank.")
    try:
        known_users = {name for name, _ in face.load_embeddings()}
        if clean_name not in known_users:
            raise HTTPException(status_code=404, detail="User is not registered.")
        return {"user_name": clean_name, "points": face.get_user_points(clean_name)}
    except HTTPException:
        raise
    except RuntimeError as error:
        logger.exception("Could not read user points")
        raise HTTPException(status_code=500, detail="Could not read user points.") from error


@app.post(
    "/api/v1/frames/process",
    tags=["webcam frames"],
    dependencies=[Depends(require_api_key)],
)
def process_frame(
    session_id: str = Form(..., min_length=1, max_length=64),
    image: UploadFile = File(..., description="A webcam frame as JPEG, PNG, or WebP."),
):
    frame = decode_image(image)
    state = get_session_state(session_id)

    with state["lock"]:
        try:
            with model_lock:
                item_model, bin_model = get_detector_models()
                detections = face.detect_litter(
                    item_model, frame, None, bin_model=bin_model
                )
                known_embeddings = face.load_embeddings()
                face_status, face_distance, facial_area = face.recognize_frame(
                    frame, known_embeddings
                )
        except RuntimeError as error:
            logger.exception("Could not load models or user data")
            raise HTTPException(
                status_code=503,
                detail="The detection service could not load its models or data.",
            ) from error
        except Exception as error:
            logger.exception("Frame processing failed")
            raise HTTPException(status_code=503, detail="Frame processing failed.") from error

        if face_status == "Scanning error":
            raise HTTPException(status_code=503, detail="Face recognition failed.")

        bin_detection = face.get_detected_bin(detections)
        drop_zone = face.get_drop_zone(frame, bin_detection)
        face.mark_detections_in_drop_zone(detections, drop_zone)
        recyclable_detections = [
            detection for detection in detections if detection["kind"] == "recyclable"
        ]

        if drop_zone is None:
            state["tracks"].clear()
            drop_detected = False
        else:
            drop_detected = face.update_drop_tracks(
                state["tracks"], recyclable_detections, drop_zone
            )

        user_name = None
        if face_status.startswith("Identified:"):
            user_name = face_status.partition(":")[2].strip()

        points_awarded = False
        points = None
        if user_name:
            try:
                points = face.get_user_points(user_name)
                now = time.monotonic()
                last_award = state["last_award_by_user"].get(
                    user_name, -face.DROP_COOLDOWN_SECONDS
                )
                if (
                    drop_detected
                    and now - last_award >= face.DROP_COOLDOWN_SECONDS
                ):
                    points = face.award_points(user_name)
                    state["last_award_by_user"][user_name] = now
                    points_awarded = True
            except (RuntimeError, ValueError) as error:
                logger.exception("Could not update user points")
                raise HTTPException(
                    status_code=500, detail="Could not update user points."
                ) from error

        return {
            "session_id": session_id,
            "face": {
                "status": face_status,
                "user_name": user_name,
                "distance": face_distance,
                "box": facial_area,
            },
            "detections": [
                {
                    **detection,
                    "box": list(detection["box"]),
                    "center": list(detection["center"]),
                }
                for detection in detections
            ],
            "selected_bin": bin_detection,
            "drop_zone": list(drop_zone) if drop_zone else None,
            "drop_detected": drop_detected,
            "points_awarded": points_awarded,
            "points": points,
        }


@app.delete(
    "/api/v1/sessions/{session_id}",
    tags=["webcam frames"],
    dependencies=[Depends(require_api_key)],
)
def reset_session(session_id: str):
    if not SESSION_ID_PATTERN.fullmatch(session_id):
        raise HTTPException(status_code=422, detail="Invalid session_id.")
    with session_lock:
        state = sessions.get(session_id)
        if state is not None:
            with state["lock"]:
                del sessions[session_id]
    return {"session_id": session_id, "reset": True}
