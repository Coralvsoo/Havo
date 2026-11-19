import csv
import subprocess
import sys
from pathlib import Path

import numpy as np

from trial_protocol import DEFAULT_PLAN, TrialRunner, classify, draw_trial_panel, load_plan


class FakeClock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now


def read_rows(path):
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def test_default_plan_has_20_drops_and_10_non_drops():
    assert len(DEFAULT_PLAN) == 30
    assert sum(spec["real_drop"] for spec in DEFAULT_PLAN) == 20


def test_classify():
    assert classify(True, True) == "TP"
    assert classify(True, False) == "FN"
    assert classify(False, True) == "FP"
    assert classify(False, False) == "TN"


def test_trial_row_records_system_output_against_plan(tmp_path):
    clock = FakeClock()
    plan = [{"trial": 1, "action": "Drop", "item": "Bottle", "real_drop": True}]
    runner = TrialRunner(plan, tmp_path, "fine-tuned", trial_seconds=5, clock=clock)
    assert runner.start_trial(video_time=12.0)
    for recognized in (True, True, False, True):
        runner.observe_frame(recognized)
    clock.now = 1.5
    runner.on_drop()
    clock.now = 1.6
    runner.on_points()
    clock.now = 4.0
    assert runner.update() is None
    clock.now = 5.0
    row = runner.update()
    assert row["drop_detected"] == "Yes" and row["points_awarded"] == "Yes"
    assert row["detection_result"] == "TP" and row["points_result"] == "TP"
    assert row["face_recognized_share"] == "0.75"
    assert row["first_drop_sec"] == "1.50" and row["video_time_start"] == "12.00"
    assert runner.done
    assert read_rows(runner.path)[0]["trial"] == "1"


def test_events_outside_a_trial_are_ignored_and_false_points_counted(tmp_path):
    clock = FakeClock()
    plan = [{"trial": 1, "action": "Floor", "item": "Bottle", "real_drop": False}]
    runner = TrialRunner(plan, tmp_path, "yolo-world", trial_seconds=2, clock=clock)
    runner.on_points()  # no trial running yet
    runner.start_trial()
    runner.on_drop()
    clock.now = 2.0
    row = runner.update()
    assert row["detection_result"] == "FP"
    assert row["points_result"] == "TN"


def test_redo_removes_last_row_and_repeats_trial(tmp_path):
    clock = FakeClock()
    runner = TrialRunner(DEFAULT_PLAN[:2], tmp_path, "fine-tuned", trial_seconds=1, clock=clock)
    runner.start_trial()
    clock.now = 1.0
    runner.update()
    assert runner.index == 1
    assert runner.redo()
    assert runner.index == 0 and read_rows(runner.path) == []
    runner.start_trial()
    assert runner.redo() and runner.active is None


def test_load_plan(tmp_path):
    path = tmp_path / "plan.csv"
    path.write_text("action,item,real_drop\nDrop,Cup,yes\nFloor,Bottle,no\n", encoding="utf-8")
    plan = load_plan(path)
    assert [spec["real_drop"] for spec in plan] == [True, False]
    assert plan[1]["trial"] == 2


def test_panel_is_drawn(tmp_path):
    runner = TrialRunner(DEFAULT_PLAN, tmp_path, "fine-tuned")
    frame = np.full((240, 320, 3), 200, dtype=np.uint8)
    draw_trial_panel(frame, runner)
    assert (frame[-5:] != 200).any() and (frame[:50] == 200).all()


def test_analyze_script(tmp_path):
    clock = FakeClock()
    plan = [
        {"trial": 1, "action": "Drop", "item": "Cup", "real_drop": True},
        {"trial": 2, "action": "Floor", "item": "Bottle", "real_drop": False},
    ]
    runner = TrialRunner(plan, tmp_path, "fine-tuned", trial_seconds=1, clock=clock)
    runner.start_trial()
    runner.on_drop()
    clock.now = 1.0
    runner.update()
    runner.start_trial()
    runner.on_drop()
    runner.on_points()
    clock.now = 2.0
    runner.update()
    script = Path(__file__).resolve().parents[1] / "scripts" / "analyze_trials.py"
    result = subprocess.run(
        [sys.executable, str(script), str(runner.path), "--out", str(tmp_path / "out")],
        capture_output=True, text=True, check=True,
    )
    assert "drop detection" in result.stdout
    summary = read_rows(tmp_path / "out" / "trials_summary.csv")
    detection = next(row for row in summary if row["level"] == "drop detection")
    points = next(row for row in summary if row["level"] == "points")
    assert (detection["TP"], detection["FP"]) == ("1", "1")
    assert (points["FN"], points["FP"]) == ("1", "1")
