"""Summarize labelled trial CSVs written by `python face.py --trials`.

Usage:
    python scripts/analyze_trials.py trials/trials_yolo-world_*.csv trials/trials_fine-tuned_*.csv

For each model it reports two levels:
- drop detection: did the system detect a drop (drop_detected), against the
  ground truth of the plan;
- points: were points awarded, which also needs a recognized face.
It also breaks the results down by action, so false points can be traced to
the actions that cause them. The tables are printed and saved as
trials_summary.csv and trials_summary.md next to the first input file
(or in --out).
"""

import argparse
import csv
from collections import OrderedDict, defaultdict
from pathlib import Path


def load_rows(paths):
    """Read all files; if a model has a trial more than once, keep the last."""
    latest = OrderedDict()
    for path in paths:
        with Path(path).open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                latest[(row["model"], int(row["trial"]))] = row
    return list(latest.values())


def metrics(results):
    counts = {name: results.count(name) for name in ("TP", "FN", "FP", "TN")}
    tp, fn, fp, tn = counts["TP"], counts["FN"], counts["FP"], counts["TN"]
    total = tp + fn + fp + tn
    precision = tp / (tp + fp) if tp + fp else None
    recall = tp / (tp + fn) if tp + fn else None
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision is not None and recall is not None and precision + recall
        else None
    )
    accuracy = (tp + tn) / total if total else None
    return {
        "trials": total,
        **counts,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": accuracy,
    }


def summarize(rows):
    by_model = defaultdict(list)
    for row in rows:
        by_model[row["model"]].append(row)
    summary = []
    for model, model_rows in by_model.items():
        shares = [
            float(row["face_recognized_share"])
            for row in model_rows
            if row.get("face_recognized_share")
        ]
        for level, column in (("drop detection", "detection_result"), ("points", "points_result")):
            entry = {"model": model, "level": level}
            entry.update(metrics([row[column] for row in model_rows]))
            entry["mean_face_share"] = sum(shares) / len(shares) if shares else None
            summary.append(entry)
    return summary


def by_action(rows):
    table = []
    groups = defaultdict(list)
    for row in rows:
        groups[(row["model"], row["action"], row["real_drop"])].append(row)
    for (model, action, real_drop), group in sorted(groups.items()):
        table.append(
            {
                "model": model,
                "action": action,
                "real_drop": real_drop,
                "trials": len(group),
                "drop_seen": sum(row["drop_detected"] == "Yes" for row in group),
                "points": sum(row["points_awarded"] == "Yes" for row in group),
            }
        )
    return table


def fmt(value):
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def markdown(rows, columns):
    lines = [
        "| " + " | ".join(columns) + " |",
        "|" + "|".join("---" for _ in columns) + "|",
    ]
    for row in rows:
        lines.append("| " + " | ".join(fmt(row[c]) for c in columns) + " |")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("files", nargs="+", help="trial CSV files")
    parser.add_argument("--out", help="output directory (default: folder of the first file)")
    args = parser.parse_args()

    rows = load_rows(args.files)
    if not rows:
        parser.error("No trials found in the given files")
    summary = summarize(rows)
    actions = by_action(rows)

    summary_columns = [
        "model", "level", "trials", "TP", "FN", "FP", "TN",
        "precision", "recall", "f1", "accuracy", "mean_face_share",
    ]
    action_columns = ["model", "action", "real_drop", "trials", "drop_seen", "points"]
    text = (
        "## Results per model\n\n"
        + markdown(summary, summary_columns)
        + "\n\n## Results per action\n\n"
        + markdown(actions, action_columns)
        + "\n"
    )
    print(text)

    out = Path(args.out) if args.out else Path(args.files[0]).parent
    out.mkdir(parents=True, exist_ok=True)
    with (out / "trials_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=summary_columns)
        writer.writeheader()
        for entry in summary:
            writer.writerow({c: fmt(entry[c]) for c in summary_columns})
    (out / "trials_summary.md").write_text(text, encoding="utf-8")
    print(f"Saved: {out / 'trials_summary.csv'} and {out / 'trials_summary.md'}")


if __name__ == "__main__":
    main()
