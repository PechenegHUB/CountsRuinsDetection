import csv
import math
from pathlib import Path

COUNTS = ("people", "hands", "cups", "dishes", "items")
FIELDS = ("row_id", *COUNTS, "occ_prob")


def read_rows(path: Path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        header = reader.fieldnames or []
        rows = list(reader)
    return header, rows


def unique_ids(rows):
    ids = [row.get("row_id") for row in rows]
    if any(not isinstance(value, str) or not value.strip() for value in ids):
        raise ValueError("Every row_id must be nonempty")
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate row_id")
    return ids


def validate(path: Path, expected_ids, allow_extra=False):
    header, rows = read_rows(path)
    if len(header) != len(FIELDS) or set(header) != set(FIELDS):
        raise ValueError("Required columns: " + ",".join(FIELDS))
    ids = unique_ids(rows)
    expected = set(expected_ids)
    missing, extra = expected - set(ids), set(ids) - expected
    if missing or (extra and not allow_extra):
        raise ValueError(f"ID mismatch: {len(missing)} missing, {len(extra)} extra")
    result = {}
    for line, row in enumerate(rows, 2):
        if None in row:
            raise ValueError(f"Line {line}: extra CSV fields")
        values = {}
        for name in (*COUNTS, "occ_prob"):
            try:
                value = float(row[name])
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Line {line}: invalid {name}") from exc
            upper = 1 if name == "occ_prob" else 50
            if not math.isfinite(value) or not 0 <= value <= upper:
                raise ValueError(f"Line {line}: {name} must be finite and in [0,{upper}]")
            values[name] = value
        result[row["row_id"]] = values
    return result


def occupancy_ap(labels, probabilities):
    if not labels:
        raise ValueError("No evaluated rows")
    positives = sum(labels)
    if positives == len(labels):
        return 1.0
    if positives == 0:
        return 1 - sum(probabilities) / len(labels)
    pairs = sorted(zip(probabilities, labels), reverse=True)
    seen = true_positives = 0
    score = 0.0
    index = 0
    while index < len(pairs):
        end = index
        newly_positive = 0
        while end < len(pairs) and pairs[end][0] == pairs[index][0]:
            newly_positive += pairs[end][1]
            end += 1
        seen += end - index
        true_positives += newly_positive
        score += newly_positive / positives * true_positives / seen
        index = end
    return score


def score(solution: Path, submission: Path):
    header, rows = read_rows(solution)
    if not set(("row_id", *COUNTS)).issubset(header):
        raise ValueError("Solution requires row_id and the five count columns")
    expected = unique_ids(rows)
    for row in rows:
        for name in COUNTS:
            value = float(row[name])
            if not math.isfinite(value) or not 0 <= value <= 50:
                raise ValueError(f"Invalid solution values for {name}")
        if "occupied" in header and float(row["occupied"]) != int(float(row["people"]) > 0):
            raise ValueError("occupied must equal (people > 0)")
        if "Usage" in header and row["Usage"] not in {"Public", "Private", "Ignored"}:
            raise ValueError("Unknown Usage value in solution")
    predictions = validate(submission, expected, allow_extra=True)
    rows = [row for row in rows if row.get("Usage") != "Ignored"]
    if not rows:
        raise ValueError("No evaluated rows")
    quality = {}
    for name in COUNTS:
        truth = [float(row[name]) for row in rows]
        if any(not math.isfinite(value) or value < 0 for value in truth):
            raise ValueError(f"Invalid solution values for {name}")
        error = sum(abs(value - predictions[row["row_id"]][name]) for value, row in zip(truth, rows))
        support = sum(truth)
        quality[name] = {
            "absolute_error": error,
            "support": support,
            "quality": max(0.0, 1 - error / (support if support else len(rows))),
        }
    labels = [float(row["people"]) > 0 for row in rows]
    probabilities = [predictions[row["row_id"]]["occ_prob"] for row in rows]
    ap = occupancy_ap(labels, probabilities)
    return {"score": .2 * ap + .8 * sum(value["quality"] for value in quality.values()) / 5,
            "rows": len(rows), "occupancy_ap": ap, "counters": quality}
