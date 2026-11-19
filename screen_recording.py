"""Record the annotated Havo window for later review."""

import csv
from datetime import datetime
from pathlib import Path
import re
import time
import uuid

import cv2


DEFAULT_OUTPUT_FPS = 10.0
MAX_REPEATED_FRAMES = 30


class ScreenRecorder:
    def __init__(
        self,
        directory,
        label,
        *,
        output_fps=DEFAULT_OUTPUT_FPS,
        clock=time.monotonic,
    ):
        self.directory = Path(directory)
        self.label = re.sub(r"[^a-zA-Z0-9_-]+", "_", str(label)).strip("_")
        self.output_fps = float(output_fps)
        if self.output_fps <= 0:
            raise ValueError("Screen recording FPS must be greater than zero")
        self.clock = clock
        self.writer = None
        self.video_path = None
        self.events_path = None
        self._events_file = None
        self._events_writer = None
        self._started_at = None
        self._frames_written = 0

    @property
    def is_recording(self):
        return self.writer is not None

    def start(self, frame_size):
        if self.is_recording:
            return self.video_path

        self.directory.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        while True:
            stem = f"screen_{self.label}_{stamp}_{uuid.uuid4().hex[:8]}"
            self.video_path = self.directory / f"{stem}.mp4"
            self.events_path = self.directory / f"{stem}_events.csv"
            if not self.video_path.exists() and not self.events_path.exists():
                break
        writer = cv2.VideoWriter(
            str(self.video_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            self.output_fps,
            tuple(int(value) for value in frame_size),
        )
        if not writer.isOpened():
            writer.release()
            if self.video_path.exists():
                self.video_path.unlink()
            raise RuntimeError(f"Could not open video writer: {self.video_path}")

        try:
            events_file = self.events_path.open(
                "w", newline="", encoding="utf-8"
            )
        except OSError:
            writer.release()
            if self.video_path.exists():
                self.video_path.unlink()
            raise

        self.writer = writer
        self._events_file = events_file
        self._events_writer = csv.writer(events_file)
        self._events_writer.writerow(
            ["video_time_sec", "event", "user", "detail"]
        )
        self._started_at = self.clock()
        self._frames_written = 0
        return self.video_path

    def elapsed(self):
        if not self.is_recording:
            return None
        return self.clock() - self._started_at

    def add_frame(self, frame):
        """Repeat frames as needed to keep the video close to real time."""
        if not self.is_recording:
            return 0
        target = int(self.elapsed() * self.output_fps) + 1
        repeats = min(
            MAX_REPEATED_FRAMES,
            max(0, target - self._frames_written),
        )
        for _ in range(repeats):
            self.writer.write(frame)
        self._frames_written += repeats
        return repeats

    def log_event(self, event, user=None, detail=""):
        if not self.is_recording:
            return
        self._events_writer.writerow(
            [f"{self.elapsed():.2f}", event, user or "", detail]
        )
        self._events_file.flush()

    def stop(self):
        if not self.is_recording:
            return None
        writer = self.writer
        events_file = self._events_file
        self.writer = None
        self._events_file = None
        self._events_writer = None
        try:
            writer.release()
        finally:
            events_file.close()
        return self.video_path

    def toggle(self, frame_size):
        if self.is_recording:
            self.stop()
            return False
        self.start(frame_size)
        return True


def draw_recording_indicator(frame, recorder):
    """Draw the REC marker on the display frame after it has been recorded."""
    if recorder is None or not recorder.is_recording:
        return
    x = frame.shape[1] - 110
    cv2.circle(frame, (x, 30), 8, (0, 0, 255), -1)
    cv2.putText(
        frame,
        f"REC {int(recorder.elapsed())}s",
        (x + 15, 36),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (0, 0, 255),
        2,
    )
