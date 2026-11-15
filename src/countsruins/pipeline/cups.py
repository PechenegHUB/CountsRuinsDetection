"""Confirm cup proposals and correct counts with neighboring frames."""
from collections import defaultdict
import json

import numpy as np
import pandas as pd
from PIL import Image
from scipy.optimize import linear_sum_assignment

from .geometry import BOX, arrays, assign, box_iou, geometry, iou, nms, pairs
from .inputs import embeddings

SCORE = {"production": .35, "student": .20, "teacher": .15, "yoloe": .15}
DFINE_CUPS = ("Cup", "Wine Glass")
DISTRACTORS = ("Bottle", "Vase", "Canned", "Flower", "Potted Plant")
YOLO_CUPS = ("cup", "mug", "drinking glass", "takeaway coffee cup")
MATCH = .30


def same_instance(a, b, overlap):
    inter = (np.minimum(a[:, None, 2:], b[None, :, 2:])
             - np.maximum(a[:, None, :2], b[None, :, :2])).clip(0).prod(2)
    smaller = np.minimum(np.prod(a[:, 2:] - a[:, :2], axis=1)[:, None],
                         np.prod(b[:, 2:] - b[:, :2], axis=1)[None, :])
    return (overlap >= MATCH) | ((smaller > 0) & (inter >= .70 * smaller))


def duplicate(box, other):
    return bool(same_instance(box, other, iou(box, other)).any())


def match(anchors, candidates):
    overlap = iou(anchors, candidates)
    eligible = same_instance(anchors, candidates, overlap)
    i, j = linear_sum_assignment(-eligible.astype(float) - .01 * overlap)
    return {int(a): int(b) for a, b in zip(i, j) if eligible[a, b]}


def deduplicate(boxes):
    distinct = []
    for box in boxes:
        if not duplicate(box[None, :], arrays(distinct)):
            distinct.append(box)
    return arrays(distinct)


def frame_detections(inputs, frame_ids):
    target_frames = set(frame_ids)
    targets = inputs.targets
    split = "test"
    assert target_frames <= set(targets.index[targets.split.eq(split)])
    excluded = set(targets.index[~targets.split.eq(split)])
    context = {f: [n for n in targets.loc[f, "context_frames"].split(";") if n and n not in excluded]
               for f in target_frames}
    all_neighbors = {n for ns in context.values() for n in ns}
    p = inputs.production[inputs.production.frame_id.isin(target_frames | all_neighbors) & inputs.production.class_name.eq("cup")]
    production = dict(tuple(p.rename(columns={"confidence": "score"}).groupby("frame_id")))
    student = inputs.grouped_boxes("SAMIR/student_pred/tables.csv", target_frames, ["cup"])
    teacher = inputs.grouped_boxes("SAMIR/dfine/tables.csv", target_frames, DFINE_CUPS + DISTRACTORS)
    near_teacher = inputs.grouped_boxes("SAMIR/dfine/context.csv", all_neighbors, DFINE_CUPS)
    near_student = inputs.grouped_boxes("SAMIR/student_pred/context.csv", all_neighbors, ["cup"])
    yoloe = defaultdict(list)
    with (inputs.cache / "submission-exp016/yoloe/detections.jsonl").open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            if row["frame_id"] in target_frames and (row["view"], row["imgsz"]) == ("expanded", 960):
                yoloe[row["frame_id"]].extend(
                    (d["confidence"], *d["xyxy"]) for d in row["detections"]
                    if d["prompt"] in YOLO_CUPS)
    return target_frames, context, production, student, teacher, near_teacher, near_student, yoloe


def instances(inputs, frame_ids):
    train, context, production, student, teacher, near_teacher, near_student, yoloe = frame_detections(inputs, frame_ids)
    empty = pd.DataFrame(columns=["score", "name", *BOX], dtype=float)
    features = []
    table_data = inputs.tables
    for f, table in table_data[table_data.frame_id.isin(train)].groupby("frame_id"):
        d = teacher.get(f, empty)
        blocked = d[d.name.isin(DISTRACTORS) & d.score.ge(.25)][BOX].to_numpy(float)
        sources = {
            "production": production.get(f, empty),
            "student": student.get(f, empty),
            "teacher": d[d.name.isin(DFINE_CUPS)],
            "yoloe": pd.DataFrame(yoloe[f], columns=["score", *BOX]),
        }
        sources = {name: frame.loc[frame.score.ge(SCORE[name]), BOX + ["score"]].to_numpy(float)
                   for name, frame in sources.items()}
        near = []
        for nf in context[f]:
            near.append(np.concatenate([
                src.loc[src.score.ge(threshold), BOX].to_numpy(float)
                for src, threshold in [(production.get(nf, empty), .35),
                                       (near_student.get(nf, empty), .20),
                                       (near_teacher.get(nf, empty), .20)]
            ]))
        # Match each detection at most once before assigning one table per instance.
        table_boxes = table[BOX].to_numpy(float)
        anchors, votes = [], []
        for name, source in sources.items():
            ordered = source[np.argsort(-source[:, 4], kind="stable"), :4]
            distinct = deduplicate(ordered)
            matched = match(arrays(anchors), distinct)
            for i in matched:
                votes[i].add(name)
            used = set(matched.values())
            for i, box in enumerate(distinct):
                if i not in used and not duplicate(box[None, :], arrays(anchors)):
                    anchors.append(box)
                    votes.append({name})
        anchors = arrays(anchors)
        repeated = [match(anchors, deduplicate(n)) for n in near]
        owners, shares = geometry(anchors, table_boxes)
        for i in np.flatnonzero(shares >= .60):
            box = anchors[i:i+1]
            repeats = sum(i in matched for matched in repeated)
            corroborated = len(votes[i]) >= 2 and ("yoloe" in votes[i] or repeats >= 1)
            confirmed = "production" in votes[i] or (
                not duplicate(box, blocked) and (corroborated or repeats >= 2))
            features.append({
                "frame_id": f, "row_id": table.row_id.iloc[owners[i]],
                **dict(zip(BOX, box[0])), **{key: key in votes[i] for key in sources},
                "repeats": repeats, "confirmed": confirmed,
            })
    return pd.DataFrame(features)


def historical(inputs):
    aux = pd.read_csv(inputs.data / "aux_boxes.csv")
    rows = []
    for f, group in aux[aux.class_name.eq("cup")].groupby("frame_id"):
        assert f.startswith("aux_") and f not in inputs.targets.index
        with Image.open(inputs.data / f"{f}.jpg") as im:
            width, height = im.size
        for row in group.itertuples():
            rows.append([f, (row.cx-row.w/2)*width, (row.cy-row.h/2)*height,
                         (row.cx+row.w/2)*width, (row.cy+row.h/2)*height])
    return pd.DataFrame(rows, columns=["frame_id", *BOX])


def fit_verifier(positive, negatives):
    from sklearn.linear_model import LogisticRegression
    x = np.vstack([positive, negatives])
    y = np.r_[np.ones(len(positive)), np.zeros(len(negatives))]
    return LogisticRegression(C=1.0, class_weight="balanced", random_state=42).fit(x, y)


def disputes(inputs):
    audit = inputs.cache / "d13"
    manifest = json.loads((audit / "cases.json").read_text(encoding="utf-8"))
    annotations = json.loads((audit / "annotations.json").read_text(encoding="utf-8"))
    assert manifest["dataset_id"] == annotations["dataset_id"]
    counts = pd.Series([r["row_id"] for r in annotations["objects"]
                        if r["class"] == "cup" and r["count"] == "yes"
                        and r["location"] == "tabletop" and r["bbox"] and not r["absent"]],
                       dtype=str).value_counts()
    assert set(counts.index) <= {case["row_id"] for case in manifest["cases"]}
    return counts.index[inputs.labels.loc[counts.index, "cups"].eq(0)]


def episodes(labels,frames,targets):
    ids=labels.frame_id.unique().tolist()
    parent={f:f for f in ids}
    def find(f):
        while parent[f]!=f:
            parent[f]=parent[parent[f]]
            f=parent[f]
        return f
    def union(a,b):
        parent[find(a)]=find(b)
    stamps=pd.to_datetime(frames.loc[ids,'timestamp'],utc=True)
    ordered=stamps.sort_values().index.tolist()
    for a,b in zip(ordered,ordered[1:]):
        if (stamps[b]-stamps[a]).total_seconds()<=300:
            union(a,b)
    seen={}
    for f in ids:
        for image in [f]+[v for v in targets.loc[f,'context_frames'].split(';') if v]:
            if image in seen:
                union(f,seen[image])
            seen[image]=f
    return {f:find(f) for f in ids}


def inside(box, table):
    x,y,u,v = box
    a,b,c,d = table
    return a <= (x+u)/2 <= c and b <= (y+v)/2 <= d


def local_boxes(group, table, minimum):
    kept = []
    for score,*box in group:
        if score >= minimum and inside(box, table) and all(box_iou(box, old)<.5 for old in kept):
            kept.append(box)
    return kept


def transfer(box, source, target):
    ax,ay,bx,by = source
    cx,cy,dx,dy = target
    sx,sy = (dx-cx)/(bx-ax),(dy-cy)/(by-ay)
    x,y,u,v = box
    return [cx+(x-ax)*sx,cy+(y-ay)*sy,cx+(u-ax)*sx,cy+(v-ay)*sy]


def recover(frame, table, neighbors, production, weak):
    strong = local_boxes(production.get(frame,()),table,.25)
    candidates = local_boxes(weak.get(frame,()),table,.10)
    added = []
    for other, other_table in neighbors:
        for box in local_boxes(production.get(other,()),other_table,.25):
            projected = transfer(box,other_table,table)
            matches = [b for b in candidates if box_iou(b,projected)>=.30
                       and all(box_iou(b,old)<.30 for old in strong+added)]
            if matches and all(box_iou(projected,old)<.30 for old in strong+added):
                added.append(max(matches,key=lambda b:box_iou(b,projected)))
    return len(strong), len(added)


def groups(df):
    return {f:g[["score",*BOX]].to_numpy(float).tolist()
            for f,g in df.sort_values("score",ascending=False).groupby("frame_id")}
def recover_cups(inputs, base, heads):
    """Verify current proposals, then recover cups supported across frames."""
    proposals = instances(inputs, inputs.test_layout.frame_id.unique())
    proposals = proposals.loc[proposals.confirmed].reset_index(drop=True)
    head, threshold = heads.get('cups')
    test_vectors = embeddings(proposals[["frame_id", *BOX]], inputs.cache / "d21/embeddings.npz", "cuda")
    verified = proposals.loc[head.predict_proba(test_vectors)[:, 1] >= threshold]
    count = verified.groupby("row_id").size().reindex(base.index, fill_value=0)
    result = base.cups + .50 * (count - base.cups).clip(lower=0)

    prod = inputs.production
    production = groups(prod[prod.class_name.eq("cup")].rename(columns={"confidence": "score"}))
    table_groups = {frame: rows[BOX].to_numpy(float).tolist()
                    for frame, rows in prod[prod.class_name.eq("table")].groupby("frame_id")}
    raw = pd.concat([inputs.boxes(f"SAMIR/dfine/{file}.csv") for file in ("tables", "context")])
    assert set(inputs.test_layout.frame_id) <= set(raw.frame_id)
    weak = groups(raw[raw.name.eq("Cup")])
    for rid, row in inputs.test_layout.set_index("row_id").iterrows():
        table = row[BOX].to_numpy(float)
        neighbors = []
        for frame in inputs.targets.loc[row.frame_id, "context_frames"].split(";"):
            if table_groups.get(frame):
                best = max(table_groups[frame], key=lambda box: box_iou(box, table))
                if box_iou(best, table) >= .65:
                    neighbors.append((frame, best))
        visible, added = recover(row.frame_id, table, neighbors, production, weak)
        if added:
            result.loc[rid] = max(result.loc[rid], visible + added)
    return base.assign(cups=result)


def reconcile_cups(inputs, base):
    """Adjust a count only when both sources agree across all three frames."""
    layout_rows = inputs.test_layout
    neighbors = {frame: [n for n in inputs.targets.loc[frame, "context_frames"].split(";") if n]
                 for frame in layout_rows.frame_id.unique()}
    frame_ids = set(neighbors) | {n for images in neighbors.values() for n in images}
    assert set(neighbors) <= set(inputs.targets.index[inputs.targets.split.eq("test")])

    student = pd.concat([inputs.boxes(f"SAMIR/student_pred/{kind}.csv", frame_ids)
                         for kind in ("tables", "context")], ignore_index=True)
    production = inputs.production[inputs.production.frame_id.isin(frame_ids)].rename(
        columns={"class_name": "name", "confidence": "score"})
    raw = {"production": production[production.name.eq("cup")],
           "student": student[student.name.eq("cup")]}
    by_frame = {key: dict(tuple(rows.groupby("frame_id"))) for key, rows in raw.items()}
    production_coverage = set(production.frame_id)
    student_coverage = set(zip(student.frame_id, student.row_id))

    corrections = {}
    for frame, layout in layout_rows.groupby("frame_id"):
        adjacent = neighbors[frame]
        dates = pd.to_datetime(inputs.frames.loc[[frame, *adjacent], "timestamp"], format="mixed", utc=True)
        if len(adjacent) != 2 or dates.dt.date.nunique() != 1:
            continue
        owned = {}
        for source, cutoff in (("production", .25), ("student", .50)):
            for image in (frame, *adjacent):
                rows = by_frame[source].get(image, raw[source].iloc[:0])
                if source == "student":
                    rows = rows[rows.row_id.isin(layout.row_id)]
                owned[image, source] = assign(nms(rows[rows.score.ge(cutoff)]), layout)
        for rid in layout.row_id:
            if not all(n in production_coverage and (n, rid) in student_coverage for n in adjacent):
                continue
            counts = []
            for source in raw:
                current, previous, following = [
                    owned[image, source].loc[lambda df: df.owner.eq(rid), BOX].to_numpy()
                    for image in (frame, *adjacent)
                ]
                n = len(current)
                if all(len(other) == n and len(pairs(current, other, .30)) == n
                       for other in (previous, following)):
                    counts.append(n)
            if len(counts) == 2 and counts[0] == counts[1]:
                corrections[rid] = int(np.sign(counts[0] - base.at[rid, "cups"]))
    adjustment = pd.Series(corrections, dtype=int).reindex(base.index, fill_value=0)
    return base.assign(cups=base.cups + adjustment)


def prepare_cup_head(inputs):
    from sklearn.model_selection import LeaveOneGroupOut
    audit = pd.read_csv(inputs.cache / "d6/instances.csv")
    training = pd.concat([audit[["frame_id", *BOX]], historical(inputs)], ignore_index=True)
    vectors = embeddings(training, inputs.cache / "d7/embeddings.npz", "cuda")
    positives = vectors[len(audit):]
    negatives = audit[audit.row_id.isin(inputs.labels.index[inputs.labels.cups.eq(0)]) &
                      ~audit.row_id.isin(disputes(inputs))]
    group_ids = negatives.frame_id.map(episodes(inputs.labels, inputs.frames, inputs.targets))
    x = vectors[negatives.index]
    oof = np.zeros(len(x))
    for keep, held in LeaveOneGroupOut().split(x, groups=group_ids):
        oof[held] = fit_verifier(positives, x[keep]).predict_proba(x[held])[:, 1]
    head = fit_verifier(positives, x)
    threshold = max(.50, float(np.nextafter(max(oof.max(), head.predict_proba(x)[:, 1].max()), 1.)))
    return head, threshold
