#!/usr/bin/env python3
"""Offline evaluation of recorded drop-detection sessions."""

import csv
import json
import sys
from pathlib import Path

import cv2

PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))
import face
from session_metrics import match_events


def _time(record):
    for key in ("time_sec", "timestamp", "received_at", "time", "at"):
        if key in record:
            try:
                return float(record[key])
            except (TypeError, ValueError):
                pass
    return None


def read_csv(path):
    if not path.is_file():
        return []
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def sensor_drop_times(session_dir):
    session_dir = Path(session_dir)
    times = []
    for name in ("sensor_drop.csv", "events.csv"):
        path = session_dir / name
        for record in read_csv(path):
            event_type = str(record.get("event_type", record.get("type", ""))).lower()
            if event_type == "sensor_drop":
                value = _time(record)
                if value is not None:
                    times.append(value)
    for name in ("sensor_drop.json", "sensor_events.json", "sensor.json", "events.json"):
        path = session_dir / name
        try:
            with path.open("r", encoding="utf-8-sig") as handle:
                data = json.load(handle)
        except FileNotFoundError:
            continue
        records = data if isinstance(data, list) else data.get("events", [])
        if isinstance(records, dict):
            records = [records]
        for record in records:
            event_type = str(record.get("event_type", record.get("type", ""))).lower()
            active = str(record.get("active", "")).lower()
            if "drop" in event_type or record.get("drop_detected") is True or record.get("sensor_drop") is True or (not event_type and active in ("1", "true", "yes")):
                value = _time(record)
                if value is not None:
                    times.append(value)
    return sorted(times)


def truth_times(session_dir):
    session_dir = Path(session_dir)
    sensor_times = sensor_drop_times(session_dir)
    config = read_session_config(session_dir)
    if config.get("sensor_connected") or sensor_times:
        return sensor_times, "sensor events"
    event_times = {
        str(record.get("event_id", "")): _time(record)
        for record in read_csv(session_dir / "events.csv")
    }
    positives = []
    for record in read_csv(session_dir / "labels.csv"):
        if str(record.get("label", "")).strip().lower() in ("1", "true", "y", "yes"):
            value = _time(record)
            if value is None:
                value = event_times.get(str(record.get("event_id", "")))
            if value is not None:
                positives.append(value)
    return sorted(positives), "human labels"


def read_session_config(session_dir):
    """Read session.json event/calibration settings without inventing defaults."""
    session_dir = Path(session_dir)
    path = session_dir / "session.json"
    if not path.is_file():
        return {}
    with path.open("r", encoding="utf-8-sig") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"session.json must contain a JSON object: {path}")
    return data


def find_recorded_segments(session_dir):
    session_dir = Path(session_dir)
    segments_dir = session_dir / "segments"
    if segments_dir.is_dir():
        return sorted(segments_dir.glob("segment_*.mp4"))
    return []


def read_fixed_zone(session_dir):
    session_dir = Path(session_dir)
    config = read_session_config(session_dir)
    zone_mode = config.get("zone_mode")
    if zone_mode not in (None, "lock", "detect", "fixed"):
        raise ValueError(
            f"Unsupported session.json zone_mode: {zone_mode!r}"
        )
    saved_zone = config.get("drop_zone")
    if isinstance(saved_zone, (list, tuple)) and len(saved_zone) == 4:
        return tuple(int(value) for value in saved_zone)
    for name in ("fixed_zone.csv", "drop_zone.csv", "zone.csv"):
        records = read_csv(session_dir / name)
        if records:
            record = records[0]
            try:
                return tuple(int(record[key]) for key in ("x1", "y1", "x2", "y2"))
            except (KeyError, ValueError):
                pass
    for name in ("fixed_zone.json", "drop_zone.json", "zone.json"):
        path = session_dir / name
        if path.is_file():
            with path.open("r", encoding="utf-8-sig") as handle:
                record = json.load(handle)
            if isinstance(record, list):
                record = record[0]
            box = record.get("box") or record.get("drop_zone")
            if isinstance(box, (list, tuple)) and len(box) == 4:
                return tuple(int(value) for value in box)
            try:
                return tuple(int(record[key]) for key in ("x1", "y1", "x2", "y2"))
            except (KeyError, ValueError):
                pass
    return None


def read_frame_timestamps(session_dir):
    rows = read_csv(Path(session_dir) / "segments" / "frames.csv")
    timestamps = {}
    for row in rows:
        try:
            timestamps[(int(row["segment"]), int(row["frame_in_segment"]))] = (
                float(row["time_sec"]),
                int(row["global_frame"]),
            )
        except (KeyError, TypeError, ValueError):
            continue
    return timestamps


def _read_box(config, key):
    box = config.get(key)
    if isinstance(box, dict):
        try:
            box = [box[name] for name in ("x1", "y1", "x2", "y2")]
        except KeyError:
            return None
    if isinstance(box, (list, tuple)) and len(box) == 4:
        return tuple(int(value) for value in box)
    return None


def replay_segment(
    clip,
    litter_model,
    bin_model,
    mode,
    *,
    segment_index,
    frame_timestamps,
    time_base,
    state,
    fixed_zone=None,
    fixed_bin_box=None,
):
    capture = cv2.VideoCapture(str(clip))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open recorded segment: {clip}")
    fps = capture.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 0:
        capture.release()
        raise RuntimeError(f"Could not read FPS from recorded segment: {clip}")

    detected_times = state["detected_times"]
    recent_above = state.setdefault("recent_above", [])
    frame_index = 0
    try:
        while True:
            success, frame = capture.read()
            if not success:
                break
            timestamp = frame_timestamps.get((segment_index, frame_index))
            time_sec = (
                timestamp[0]
                if timestamp is not None
                else time_base + frame_index / fps
            )
            frame_index += 1

            items = []
            calibrating = (
                mode == "lock"
                and state["locked_box"] is None
                and state["calibration_count"] < face.CALIBRATION_FRAMES
            )
            bin_detection = None
            if mode == "lock":
                if calibrating:
                    state["calibration_boxes"].append(
                        [item["box"] for item in face.detect_bins(bin_model, frame)]
                    )
                    state["calibration_count"] += 1
                    if state["calibration_count"] >= face.CALIBRATION_FRAMES:
                        state["locked_box"], _reason = face.decide_calibration(
                            state["calibration_boxes"]
                        )
                        if state["locked_box"] is None:
                            state["locked_box"] = fixed_bin_box
                if state["locked_box"] is not None:
                    if not calibrating:
                        bins = face.detect_bins(bin_model, frame)
                        bin_match = face.match_bin_detection(
                            bins, state["locked_box"]
                        )
                        state["drift"] = face.update_drift(
                            state["drift"],
                            1.0 if bin_match is not None else None,
                        )
                        if bin_match is not None and not state["drift"]["drifted"]:
                            state["locked_box"] = bin_match["box"]
                    bin_detection = {
                        "box": state["locked_box"],
                        "label": "calibrated bin",
                    }
            elif mode == "detect":
                bins = face.detect_bins(bin_model, frame)
                bin_detection = (
                    max(bins, key=lambda detection: detection["confidence"])
                    if bins
                    else None
                )
            elif mode == "fixed":
                if fixed_bin_box is not None:
                    bin_detection = {"box": fixed_bin_box, "label": "fixed bin"}

            if mode == "fixed" and bin_detection is None:
                drop_zone = fixed_zone
            else:
                drop_zone = face.get_drop_zone(frame, bin_detection)

            bin_height = (
                bin_detection["box"][3] - bin_detection["box"][1]
                if bin_detection is not None
                else None
            )
            if (
                mode != "lock"
                or not calibrating
                or state["locked_box"] is not None
            ):
                items = face.detect_items(litter_model, frame, drop_zone)

            face.mark_detections_in_drop_zone(items, drop_zone)
            if drop_zone is None or (
                mode == "lock" and state["drift"]["drifted"]
            ):
                state["tracks"].clear()
                recent_above.clear()
            elif face.update_drop_tracks(
                state["tracks"],
                items,
                drop_zone,
                bin_height,
                recent_above,
            ):
                detected_times.append(time_sec)
    finally:
        capture.release()
    return frame_index / fps


def replay_session(session_dir, litter_model, bin_model, mode):
    clips = find_recorded_segments(session_dir)
    if not clips:
        raise RuntimeError(f"No recorded MP4 segments found in {session_dir}")
    config = read_session_config(session_dir)
    fixed_bin_box = _read_box(config, "calibrated_bin_box")
    fixed_zone = read_fixed_zone(session_dir)
    if mode == "fixed" and fixed_zone is None and fixed_bin_box is None:
        raise RuntimeError(
            f"Fixed-zone mode requires session.json calibrated_bin_box "
            f"or drop_zone in {session_dir}"
        )
    state = {
        "detected_times": [],
        "tracks": {},
        "recent_above": [],
        "locked_box": None,
        "calibration_boxes": [],
        "calibration_count": 0,
        "drift": {"bad_checks": 0, "drifted": False},
    }
    frame_timestamps = read_frame_timestamps(session_dir)
    time_base = 0.0
    for clip in clips:
        segment_index = int(clip.stem.rsplit("_", 1)[-1])
        segment_duration = replay_segment(
            clip,
            litter_model,
            bin_model,
            mode,
            segment_index=segment_index,
            frame_timestamps=frame_timestamps,
            time_base=time_base,
            state=state,
            fixed_zone=fixed_zone,
            fixed_bin_box=fixed_bin_box,
        )
        time_base += segment_duration
    return sorted(state["detected_times"])


def calculate_metrics(detected_times, true_times, tolerance):
    matches, unmatched_detected, unmatched_true = match_events(
        detected_times, true_times, tolerance
    )
    tp = len(matches)
    fp = len(unmatched_detected)
    fn = len(unmatched_true)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    delays = [
        float(detected_times[detected_index]) - float(true_times[true_index])
        for detected_index, true_index, _difference in matches
    ]
    mean_delay = sum(delays) / len(delays) if delays else None
    return {
        "TP": tp,
        "FP": fp,
        "FN": fn,
        "precision": precision,
        "recall": recall,
        "F1": f1,
        "mean_detection_delay": mean_delay,
    }


def write_evaluation_csv(path, rows):
    fields = ["session", "mode", "TP", "FP", "FN", "precision", "recall", "F1", "mean_detection_delay"]
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_evaluation_markdown(path, rows):
    fields = ["session", "mode", "TP", "FP", "FN", "precision", "recall", "F1", "mean_detection_delay"]
    with path.open("w", encoding="utf-8") as handle:
        handle.write("| " + " | ".join(fields) + " |\n")
        handle.write("| " + " | ".join("---" for _ in fields) + " |\n")
        for row in rows:
            handle.write("| " + " | ".join(str(row[field]) for field in fields) + " |\n")


def evaluate_sessions(session_dirs, tolerance=1.0):
    print("Loading fine-tuned bin detector and YOLO-World item detector...")
    litter_model, bin_model = face.load_litter_models(face.MODEL_FINE_TUNED)
    print("Model source: fine-tuned bin + YOLO-World recyclable items")

    rows = []
    for session_dir in session_dirs:
        session_dir = Path(session_dir)
        if not session_dir.is_dir():
            raise NotADirectoryError(f"Session directory not found: {session_dir}")
        truth, truth_source = truth_times(session_dir)
        print(f"Ground truth source: {truth_source}")
        for mode in ("lock", "detect", "fixed"):
            print(f"Evaluating {session_dir} mode={mode}")
            detected = replay_session(session_dir, litter_model, bin_model, mode)
            metrics = calculate_metrics(detected, truth, tolerance)
            row = {"session": str(session_dir), "mode": mode, **metrics}
            rows.append(row)
            print(
                f"{mode}: TP={metrics['TP']} FP={metrics['FP']} FN={metrics['FN']} "
                f"precision={metrics['precision']:.3f} recall={metrics['recall']:.3f} "
                f"F1={metrics['F1']:.3f} mean_detection_delay={metrics['mean_detection_delay']}"
            )
    return rows


def main(argv=None):
    import argparse
    parser = argparse.ArgumentParser(description="Evaluate recorded sessions offline")
    parser.add_argument("session_dir", metavar="SESSION_DIR", nargs="?")
    parser.add_argument("--sessions", nargs="*", metavar="DIR")
    parser.add_argument("--tolerance", type=float, default=1.0)
    args = parser.parse_args(argv)
    session_dirs = ([args.session_dir] if args.session_dir else []) + (
        args.sessions or []
    )
    if not session_dirs:
        parser.error("provide SESSION_DIR or one or more --sessions directories")
    if args.tolerance < 0:
        parser.error("--tolerance must be non-negative")
    try:
        rows = evaluate_sessions(session_dirs, args.tolerance)
        output_dir = Path(session_dirs[0]).resolve()
        csv_path = output_dir / "evaluation.csv"
        md_path = output_dir / "evaluation.md"
        write_evaluation_csv(csv_path, rows)
        write_evaluation_markdown(md_path, rows)
        print(f"Wrote {csv_path}")
        print(f"Wrote {md_path}")
    except (OSError, RuntimeError, ValueError, cv2.error) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
