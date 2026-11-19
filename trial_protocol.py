"""Scripted trials with automatic, labelled logging.

In trial mode the app walks the participant through a fixed plan of trials.
Each trial has a known ground truth (whether the action is a real drop into
the bin), taken from the plan. The participant presses SPACE, performs the
action shown on screen, and the app records what the system did during a
fixed time window: whether a drop was detected, whether points were awarded,
and how often the face was recognized. One labelled row per trial is written
to a CSV file, which scripts/analyze_trials.py turns into metrics.
"""

import csv
from datetime import datetime
from pathlib import Path
import re
import time

import cv2


TRIAL_SECONDS = 8.0

# Same order as the pilot-test sheet (shuffled once with seed 42).
_DEFAULT_ORDER = [
    ("Normal / fast drop into the bin", "Cup", True),
    ("Normal / fast drop into the bin", "Bottle", True),
    ("Normal / fast drop into the bin", "Bottle", True),
    ("Drop it on the floor next to the bin", "Bottle", False),
    ("Hold above the bin 3 s, then take it away", "Bottle", False),
    ("Slow drop into the bin", "Cup", True),
    ("Slow drop into the bin", "Cup", True),
    ("Normal / fast drop into the bin", "Bottle", True),
    ("Normal / fast drop into the bin", "Bottle", True),
    ("Normal / fast drop into the bin", "Cup", True),
    ("Slow drop into the bin", "Cup", True),
    ("Move it past the bin without dropping", "Bottle", False),
    ("Hand into the drop zone, no item", "None", False),
    ("Hold above the bin 3 s, then take it away", "Bottle", False),
    ("Normal / fast drop into the bin", "Cup", True),
    ("Drop it on the floor next to the bin", "Bottle", False),
    ("Slow drop into the bin", "Bottle", True),
    ("Normal / fast drop into the bin", "Bottle", True),
    ("Normal / fast drop into the bin", "Cup", True),
    ("Slow drop into the bin", "Bottle", True),
    ("Normal / fast drop into the bin", "Cup", True),
    ("Hand into the drop zone, no item", "None", False),
    ("Slow drop into the bin", "Bottle", True),
    ("Move it past the bin without dropping", "Bottle", False),
    ("Slow drop into the bin", "Cup", True),
    ("Slow drop into the bin", "Cup", True),
    ("Move it past the bin without dropping", "Bottle", False),
    ("Slow drop into the bin", "Bottle", True),
    ("Slow drop into the bin", "Bottle", True),
    ("Hold above the bin 3 s, then take it away", "Bottle", False),
]

DEFAULT_PLAN = [
    {"trial": index, "action": action, "item": item, "real_drop": real_drop}
    for index, (action, item, real_drop) in enumerate(_DEFAULT_ORDER, start=1)
]

FIELDS = [
    "model",
    "trial",
    "action",
    "item",
    "real_drop",
    "drop_detected",
    "drops_detected_count",
    "points_awarded",
    "detection_result",
    "points_result",
    "face_recognized_share",
    "first_drop_sec",
    "first_points_sec",
    "video_time_start",
    "started_at",
]


def _yes(value):
    return "Yes" if value else "No"


def parse_bool(text):
    value = str(text).strip().lower()
    if value in {"yes", "y", "true", "1"}:
        return True
    if value in {"no", "n", "false", "0"}:
        return False
    raise ValueError(f"Expected yes/no, got {text!r}")


def classify(real, predicted):
    if real:
        return "TP" if predicted else "FN"
    return "FP" if predicted else "TN"


def load_plan(path):
    """Read a plan CSV with columns action, item, real_drop (yes/no)."""
    with Path(path).open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"Trial plan is empty: {path}")
    return [
        {
            "trial": index,
            "action": row["action"].strip(),
            "item": row.get("item", "").strip(),
            "real_drop": parse_bool(row["real_drop"]),
        }
        for index, row in enumerate(rows, start=1)
    ]


class TrialRunner:
    def __init__(
        self,
        plan,
        directory,
        model_label,
        *,
        trial_seconds=TRIAL_SECONDS,
        clock=time.monotonic,
    ):
        if not plan:
            raise ValueError("Trial plan is empty")
        if trial_seconds <= 0:
            raise ValueError("Trial length must be greater than zero")
        self.plan = list(plan)
        self.model_label = model_label
        self.trial_seconds = float(trial_seconds)
        self.clock = clock
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        safe_label = re.sub(r"[^a-zA-Z0-9_-]+", "_", model_label).strip("_")
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.path = self.directory / f"trials_{safe_label}_{stamp}.csv"
        self.rows = []
        self.active = None
        self.last_row = None
        self._write()

    @property
    def index(self):
        return len(self.rows)

    @property
    def done(self):
        return self.index >= len(self.plan) and self.active is None

    @property
    def current(self):
        return None if self.index >= len(self.plan) else self.plan[self.index]

    def start_trial(self, video_time=None):
        if self.active is not None or self.current is None:
            return False
        self.active = {
            "started": self.clock(),
            "started_at": datetime.now().isoformat(timespec="seconds"),
            "video_time_start": video_time,
            "drops": 0,
            "points": False,
            "first_drop": None,
            "first_points": None,
            "frames": 0,
            "face_frames": 0,
        }
        return True

    def elapsed(self):
        if self.active is None:
            return None
        return self.clock() - self.active["started"]

    def observe_frame(self, face_recognized):
        if self.active is None:
            return
        self.active["frames"] += 1
        self.active["face_frames"] += int(bool(face_recognized))

    def on_drop(self):
        if self.active is None:
            return
        self.active["drops"] += 1
        if self.active["first_drop"] is None:
            self.active["first_drop"] = self.elapsed()

    def on_points(self):
        if self.active is None:
            return
        self.active["points"] = True
        if self.active["first_points"] is None:
            self.active["first_points"] = self.elapsed()

    def update(self):
        """Close the active trial once its window has passed. Returns the row."""
        if self.active is None or self.elapsed() < self.trial_seconds:
            return None
        return self._finish()

    def redo(self):
        """Cancel the running trial, or remove the last saved one to repeat it."""
        if self.active is not None:
            self.active = None
            return True
        if self.rows:
            self.rows.pop()
            self.last_row = None
            self._write()
            return True
        return False

    def _finish(self):
        spec = self.current
        active = self.active
        drop_detected = active["drops"] > 0
        row = {
            "model": self.model_label,
            "trial": spec["trial"],
            "action": spec["action"],
            "item": spec["item"],
            "real_drop": _yes(spec["real_drop"]),
            "drop_detected": _yes(drop_detected),
            "drops_detected_count": active["drops"],
            "points_awarded": _yes(active["points"]),
            "detection_result": classify(spec["real_drop"], drop_detected),
            "points_result": classify(spec["real_drop"], active["points"]),
            "face_recognized_share": (
                f"{active['face_frames'] / active['frames']:.2f}"
                if active["frames"]
                else ""
            ),
            "first_drop_sec": (
                f"{active['first_drop']:.2f}" if active["first_drop"] is not None else ""
            ),
            "first_points_sec": (
                f"{active['first_points']:.2f}"
                if active["first_points"] is not None
                else ""
            ),
            "video_time_start": (
                f"{active['video_time_start']:.2f}"
                if active["video_time_start"] is not None
                else ""
            ),
            "started_at": active["started_at"],
        }
        self.rows.append(row)
        self.last_row = row
        self.active = None
        self._write()
        return row

    def _write(self):
        with self.path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=FIELDS)
            writer.writeheader()
            writer.writerows(self.rows)

    def status_lines(self):
        total = len(self.plan)
        if self.active is not None:
            spec = self.current
            left = max(0.0, self.trial_seconds - self.elapsed())
            return [
                f"Trial {spec['trial']}/{total}: {spec['action']} ({spec['item']})",
                f"RECORDING TRIAL - {left:.1f} s left   (N: cancel)",
            ]
        if self.current is None:
            return [
                f"All {total} trials done.",
                f"Saved: {self.path.name}   (Q: quit, N: redo last)",
            ]
        spec = self.current
        lines = [
            f"Trial {spec['trial']}/{total}: {spec['action']} ({spec['item']})",
            "Press SPACE, then do the action   (N: redo last)",
        ]
        if self.last_row is not None:
            last = self.last_row
            lines.append(
                f"Last trial {last['trial']}: drop seen={last['drop_detected']}, "
                f"points={last['points_awarded']}"
            )
        return lines


def draw_trial_panel(frame, runner):
    if runner is None:
        return
    lines = runner.status_lines()
    height = frame.shape[0]
    line_height = 22
    top = height - 12 - line_height * len(lines)
    # Translucent band, so the scene under it stays visible in the recording.
    band = frame[max(0, top - 8):height]
    band[:] = (band * 0.35).astype(band.dtype)
    for number, text in enumerate(lines):
        color = (0, 0, 255) if "RECORDING" in text else (255, 255, 255)
        cv2.putText(
            frame,
            text,
            (12, top + 14 + number * line_height),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            1,
        )
