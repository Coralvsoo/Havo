#!/usr/bin/env python3
"""Interactive review utility for recorded session clips."""

import csv
import sys
from pathlib import Path

import cv2

PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))


def _parse_time(record):
    for key in ("time_sec", "timestamp", "received_at", "time", "at"):
        if key in record:
            try:
                return float(record[key])
            except (TypeError, ValueError):
                continue
    return None


def read_csv_records(path):
    if not path.is_file():
        return []
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def read_review_events(session_dir):
    """Read event candidates from events.csv or standalone event CSV files."""
    session_dir = Path(session_dir)
    events = []
    event_records = read_csv_records(session_dir / "events.csv")
    if not event_records:
        for event_type in ("drop_detected", "zone_activity"):
            event_records.extend(
                {
                    **record,
                    "event_type": event_type,
                    "event_id": str(record.get("event_id") or record.get("id") or ""),
                }
                for record in read_csv_records(session_dir / f"{event_type}.csv")
            )
    for record in event_records:
        event_type = str(record.get("event_type", "")).strip()
        if event_type not in ("drop_detected", "zone_activity"):
            continue
        time_sec = _parse_time(record)
        event_id = str(record.get("event_id") or record.get("id") or "")
        if not event_id:
            continue
        events.append({
            "event_id": event_id,
            "event_type": event_type,
            "time_sec": time_sec,
            "record": record,
        })
    return sorted(
        events,
        key=lambda event: (
            event["event_type"] != "drop_detected",
            event["time_sec"] if event["time_sec"] is not None else float("inf"),
        ),
    )


def read_labels(path):
    labels = {}
    for record in read_csv_records(path):
        event_id = str(record.get("event_id") or record.get("id") or "")
        if not event_id:
            continue
        value = str(record.get("label", "")).strip().lower()
        labels[event_id] = "" if value in ("", "skip") else ("1" if value in ("1", "true", "y", "yes") else "0")
    return labels


def write_labels(path, events, labels):
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=["event_id", "event_type", "time_sec", "label"])
        writer.writeheader()
        for event in events:
            writer.writerow({
                "event_id": event["event_id"],
                "event_type": event["event_type"],
                "time_sec": "" if event["time_sec"] is None else f"{event['time_sec']:.6f}",
                "label": labels.get(event["event_id"], ""),
            })


def find_clip(session_dir, event):
    """Find a clip named for a candidate or return the first/only clip."""
    clips_dir = Path(session_dir) / "clips"
    clips = sorted(clips_dir.glob("*.mp4")) if clips_dir.is_dir() else []
    record = event.get("record", {})
    for key in ("clip", "clip_name", "filename", "video"):
        if record.get(key):
            requested = clips_dir / str(record[key])
            if requested.is_file():
                return requested
    event_id = str(event["event_id"])
    for clip in clips:
        if event_id in clip.stem:
            return clip
    if len(clips) == 1:
        return clips[0]
    return None


def _prompt_key():
    key = input("[Y] real  [N] no drop  [S] skip  [B] back\n> ").strip().lower()
    return key


def review_clip(clip, event, index, total):
    capture = cv2.VideoCapture(str(clip))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open clip: {clip}")
    try:
        failed_read = False
        while True:
            success, frame = capture.read()
            if not success:
                if failed_read:
                    raise RuntimeError(f"Clip contains no readable frames: {clip}")
                failed_read = True
                capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
            failed_read = False
            cv2.putText(
                frame,
                f"{index}/{total} {event['event_type']} at {event['time_sec']}",
                (12, 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 255, 255),
                2,
            )
            cv2.putText(
                frame,
                "Y real | N no drop | S skip | B back",
                (12, 56),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 255),
                2,
            )
            cv2.imshow("Review session event", frame)
            key = cv2.waitKey(30) & 0xFF
            if key in (
                ord("y"),
                ord("Y"),
                ord("n"),
                ord("N"),
                ord("s"),
                ord("S"),
                ord("b"),
                ord("B"),
            ):
                return chr(key).lower()
    finally:
        capture.release()


def review_session(session_dir):
    session_dir = Path(session_dir)
    if not session_dir.is_dir():
        raise NotADirectoryError(f"Session directory not found: {session_dir}")
    events = read_review_events(session_dir)
    labels = read_labels(session_dir / "labels.csv")
    index = next(
        (
            candidate
            for candidate, event in enumerate(events)
            if labels.get(event["event_id"], "") == ""
        ),
        len(events),
    )
    try:
        while 0 <= index < len(events):
            event = events[index]
            print(
                f"\nEvent {index + 1}/{len(events)}: id={event['event_id']} "
                f"type={event['event_type']} time={event['time_sec']}"
            )
            clip = find_clip(session_dir, event)
            if clip is not None:
                key = review_clip(clip, event, index + 1, len(events))
            else:
                print("Warning: no matching clip found.")
                key = _prompt_key()
            if key == "b":
                index = max(0, index - 1)
                continue
            if key == "s":
                labels[event["event_id"]] = ""
            elif key == "y":
                labels[event["event_id"]] = "1"
            elif key == "n":
                labels[event["event_id"]] = "0"
            else:
                print("Use Y, N, S, or B.")
                continue
            write_labels(session_dir / "labels.csv", events, labels)
            index += 1
            while (
                index < len(events)
                and labels.get(events[index]["event_id"], "") != ""
            ):
                index += 1
    finally:
        cv2.destroyAllWindows()
    print(f"Wrote {session_dir / 'labels.csv'}")


def main(argv=None):
    import argparse
    parser = argparse.ArgumentParser(description="Review detected session events")
    parser.add_argument("session_dir", metavar="SESSION_DIR")
    args = parser.parse_args(argv)
    try:
        review_session(args.session_dir)
    except (OSError, RuntimeError, csv.Error) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
