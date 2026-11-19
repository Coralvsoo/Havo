"""Pure helpers shared by session review and live/drop reporting."""


def match_events(detected_times, true_times, tolerance):
    """Greedy one-to-one matching by smallest absolute time difference."""
    candidates = sorted(
        (
            abs(float(detected) - float(true)),
            detected_index,
            true_index,
        )
        for detected_index, detected in enumerate(detected_times)
        for true_index, true in enumerate(true_times)
        if abs(float(detected) - float(true)) <= tolerance
    )
    matched_detected = set()
    matched_true = set()
    matches = []
    for difference, detected_index, true_index in candidates:
        if detected_index not in matched_detected and true_index not in matched_true:
            matches.append((detected_index, true_index, difference))
            matched_detected.add(detected_index)
            matched_true.add(true_index)
    unmatched_detected = [
        index for index in range(len(detected_times)) if index not in matched_detected
    ]
    unmatched_true = [
        index for index in range(len(true_times)) if index not in matched_true
    ]
    return matches, unmatched_detected, unmatched_true
