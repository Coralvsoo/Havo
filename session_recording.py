import csv
from collections import deque
from datetime import datetime
import json
import logging
from pathlib import Path
import time

import cv2
import numpy as np


LOGGER = logging.getLogger(__name__)


def blur_faces(frame, cascade):
    blurred = frame.copy()
    gray = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)
    faces = cascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
    )
    for x, y, width, height in faces:
        region = blurred[y : y + height, x : x + width]
        blurred[y : y + height, x : x + width] = cv2.GaussianBlur(
            region, (31, 31), 0
        )
    return blurred


class SessionRecorder:
    def __init__(
        self,
        directory,
        frame_size,
        fps,
        metadata,
        *,
        max_storage_bytes,
        blur=True,
        clip_seconds_before=3.0,
        clip_seconds_after=3.0,
        segment_minutes=5,
        clock=time.monotonic,
        video_writer_factory=None,
        face_cascade=None,
    ):
        if fps <= 0:
            raise ValueError("Recording FPS must be greater than zero")
        if min(frame_size) <= 0:
            raise ValueError("Recording frame dimensions must be positive")
        if max_storage_bytes <= 0:
            raise ValueError("Maximum recording storage must be greater than zero")
        if segment_minutes <= 0:
            raise ValueError("Segment duration must be greater than zero")
        if clip_seconds_before < 0 or clip_seconds_after < 0:
            raise ValueError("Clip durations cannot be negative")

        base_directory = Path(directory)
        base_directory.mkdir(parents=True, exist_ok=True)
        session_name = datetime.now().strftime("%Y%m%d_%H%M%S")
        session_directory = base_directory / session_name
        suffix = 1
        while session_directory.exists():
            session_directory = base_directory / f"{session_name}_{suffix:02d}"
            suffix += 1

        self.directory = session_directory
        self.segment_directory = session_directory / "segments"
        self.clip_directory = session_directory / "clips"
        self.segment_directory.mkdir(parents=True)
        self.clip_directory.mkdir(parents=True)
        self.frame_size = tuple(frame_size)
        self.fps = float(fps)
        self.max_storage_bytes = max_storage_bytes
        self.clip_seconds_before = clip_seconds_before
        self.clip_seconds_after = clip_seconds_after
        self.segment_frame_limit = max(1, int(fps * 60 * segment_minutes))
        self.clock = clock
        self.started_at = clock()
        self.video_writer_factory = video_writer_factory or cv2.VideoWriter
        self.face_cascade = face_cascade
        if blur and self.face_cascade is None:
            cascade_path = (
                Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
            )
            self.face_cascade = cv2.CascadeClassifier(str(cascade_path))
            if self.face_cascade.empty():
                raise RuntimeError(f"Could not load face-blur cascade: {cascade_path}")
        self.blur_enabled = blur
        self.ring_buffer = deque(
            maxlen=max(1, int(fps * (clip_seconds_before + clip_seconds_after + 1)))
        )
        self.pending_clips = []
        self.frame_count = 0
        self.event_count = 0
        self.segment_index = -1
        self.segment_frame_count = 0
        self.segment_writer = None
        self.closed = False
        self._frames_file = (self.segment_directory / "frames.csv").open(
            "w", newline="", encoding="utf-8"
        )
        self._frames_csv = csv.writer(self._frames_file)
        self._frames_csv.writerow(
            ["segment", "frame_in_segment", "global_frame", "time_sec"]
        )
        self._events_file = (self.directory / "events.csv").open(
            "w", newline="", encoding="utf-8"
        )
        self._events_csv = csv.writer(self._events_file)
        self._events_csv.writerow(
            [
                "event_id",
                "event_type",
                "time_sec",
                "global_frame",
                "zone_x1",
                "zone_y1",
                "zone_x2",
                "zone_y2",
                "zone_mode",
                "user",
            ]
        )
        session_metadata = dict(metadata)
        session_metadata.update(
            {
                "start_time": datetime.now().astimezone().isoformat(),
                "camera_resolution": {
                    "width": self.frame_size[0],
                    "height": self.frame_size[1],
                },
                "fps": self.fps,
                "face_blurring": self.blur_enabled,
            }
        )
        (self.directory / "session.json").write_text(
            json.dumps(session_metadata, indent=2), encoding="utf-8"
        )

    @property
    def time_sec(self):
        return max(0.0, self.clock() - self.started_at)

    def add_frame(self, frame):
        if self.closed:
            raise RuntimeError("Cannot add a frame to a closed recording")
        if frame.shape[1::-1] != self.frame_size:
            raise ValueError(
                f"Camera frame size changed from {self.frame_size} "
                f"to {frame.shape[1::-1]}"
            )
        timestamp = self.time_sec
        saved_frame = (
            blur_faces(frame, self.face_cascade) if self.blur_enabled else frame.copy()
        )
        if self.segment_writer is None or (
            self.segment_frame_count >= self.segment_frame_limit
        ):
            self._close_segment()
            self._open_segment()
        self.segment_writer.write(saved_frame)
        self._frames_csv.writerow(
            [
                self.segment_index,
                self.segment_frame_count,
                self.frame_count,
                f"{timestamp:.6f}",
            ]
        )
        self._frames_file.flush()
        entry = (self.frame_count, timestamp, saved_frame.copy())
        self.ring_buffer.append(entry)
        for pending in self.pending_clips:
            if timestamp > pending["last_frame_time"]:
                pending["frames"].append(entry)
                pending["last_frame_time"] = timestamp
        self.frame_count += 1
        self.segment_frame_count += 1
        self._flush_ready_clips(timestamp)
        return {"global_frame": self.frame_count - 1, "time_sec": timestamp}

    def record_event(
        self,
        event_type,
        *,
        time_sec=None,
        global_frame=None,
        zone=None,
        zone_mode=None,
        user=None,
        item_labels=(),
    ):
        if self.closed:
            raise RuntimeError("Cannot record an event in a closed session")
        event_time = self.time_sec if time_sec is None else float(time_sec)
        if global_frame is None:
            global_frame = max(0, self.frame_count - 1)
        self.event_count += 1
        event_id = f"event_{self.event_count:06d}"
        normalized_zone = tuple(zone) if zone is not None else None
        coordinates = normalized_zone or (None, None, None, None)
        self._events_csv.writerow(
            [
                event_id,
                event_type,
                f"{event_time:.6f}",
                global_frame,
                *coordinates,
                zone_mode,
                user,
            ]
        )
        self._events_file.flush()
        event_metadata = {
            "event_id": event_id,
            "event_type": event_type,
            "time_sec": event_time,
            "global_frame": global_frame,
            "zone": list(normalized_zone) if normalized_zone else None,
            "zone_mode": zone_mode,
            "user": user,
            "item_labels": list(item_labels),
        }
        recent_frames = [
            entry
            for entry in self.ring_buffer
            if event_time - self.clip_seconds_before <= entry[1] <= event_time
        ]
        pending = {
            "metadata": event_metadata,
            "frames": list(recent_frames),
            "last_frame_time": recent_frames[-1][1] if recent_frames else float("-inf"),
            "end_time": event_time + self.clip_seconds_after,
        }
        self.pending_clips.append(pending)
        self._flush_ready_clips(self.time_sec)
        return event_id

    def _open_segment(self):
        self.segment_index += 1
        self.segment_frame_count = 0
        segment_path = self.segment_directory / f"segment_{self.segment_index:03d}.mp4"
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self.segment_writer = self.video_writer_factory(
            str(segment_path), fourcc, self.fps, self.frame_size
        )
        if hasattr(self.segment_writer, "isOpened") and not self.segment_writer.isOpened():
            self.segment_writer.release()
            self.segment_writer = None
            raise RuntimeError(f"Could not create recording segment: {segment_path}")

    def _close_segment(self):
        if self.segment_writer is not None:
            self.segment_writer.release()
            self.segment_writer = None
            self._enforce_storage_limit()

    def _flush_ready_clips(self, timestamp):
        ready = [
            pending
            for pending in self.pending_clips
            if timestamp >= pending["end_time"]
        ]
        for pending in ready:
            self._write_clip(pending)
            self.pending_clips.remove(pending)

    def _write_clip(self, pending):
        metadata = pending["metadata"]
        clip_path = self.clip_directory / f"{metadata['event_id']}.mp4"
        frames = pending["frames"]
        if frames:
            writer = self.video_writer_factory(
                str(clip_path),
                cv2.VideoWriter_fourcc(*"mp4v"),
                self.fps,
                self.frame_size,
            )
            if hasattr(writer, "isOpened") and not writer.isOpened():
                writer.release()
                raise RuntimeError(f"Could not create event clip: {clip_path}")
            try:
                for _, _, frame in frames:
                    writer.write(frame)
            finally:
                writer.release()
        metadata_path = self.clip_directory / f"{metadata['event_id']}.json"
        metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    def _enforce_storage_limit(self):
        total_size = sum(
            path.stat().st_size for path in self.directory.rglob("*") if path.is_file()
        )
        segment_paths = sorted(self.segment_directory.glob("segment_*.mp4"))
        while total_size > self.max_storage_bytes and segment_paths:
            oldest_segment = segment_paths.pop(0)
            segment_size = oldest_segment.stat().st_size
            oldest_segment.unlink()
            total_size -= segment_size
            LOGGER.warning(
                "Recording storage limit exceeded; removed oldest segment %s",
                oldest_segment,
            )
        if total_size > self.max_storage_bytes:
            LOGGER.warning(
                "Recording exceeds storage limit, but clips and CSV files were preserved"
            )

    def close(self):
        if self.closed:
            return
        self._close_segment()
        self._frames_file.close()
        self._events_file.close()
        for pending in self.pending_clips:
            self._write_clip(pending)
        self.pending_clips.clear()
        self._enforce_storage_limit()
        self.closed = True
