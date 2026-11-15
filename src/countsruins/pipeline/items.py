"""Rank distinct item proposals against manually annotated training boxes."""
import json

import numpy as np
import pandas as pd

from .geometry import BOX, assign, iou, nms, pairs
from .inputs import embeddings

ITEM_CLASSES = ("Cell Phone", "Telephone", "Handbag/Satchel", "Backpack", "Wallet/Purse",
                "Book", "Notepaper", "Folder", "Head Phone", "earphone", "Camera", "Glasses", "Key")
WEAK_CLASSES = ("Book", "Cell Phone")
BAG_CLASSES = ("Handbag/Satchel", "Backpack")


def candidates(raw, tables):
    raw = raw[raw.frame_id.isin(tables.frame_id) & raw.name.isin(ITEM_CLASSES) & raw.score.ge(.15)]
    rows = []
    for frame, group in raw.groupby("frame_id"):
        local = assign(nms(group), tables[tables.frame_id.eq(frame)]).dropna(subset=["owner"])
        strong = local[local.score.ge(.45) & ~local.name.isin(BAG_CLASSES)]
        weak = local[local.score.lt(.45) & local.name.isin(WEAK_CLASSES)]
        distinct = iou(weak[BOX].to_numpy(float), strong[BOX].to_numpy(float)).max(axis=1, initial=0) < .30
        rows.append(weak.loc[distinct])
    return pd.concat(rows, ignore_index=True)


def fit(x,y):
    from sklearn.linear_model import LogisticRegression
    return LogisticRegression(C=1,class_weight="balanced",random_state=419).fit(x,y)


def gate(scores,y):
    return float(np.nextafter(scores[y==0].max(),np.inf))


def box_evidence(pool, x, student, layout_rows):
    support = np.zeros(len(pool))
    clean = {}
    for frame, layout in layout_rows.groupby("frame_id"):
        l = assign(nms(student[student.frame_id.eq(frame) & student.score.ge(.20)]), layout)
        raw = x[x.frame_id.eq(frame) & x.score.ge(.15)]
        strong = nms(raw)
        strong = strong[strong.score.ge(.45) & ~strong.name.isin(BAG_CLASSES)]
        owned = assign(strong, layout)
        for rid in layout.row_id:
            local = pool[pool.owner.eq(rid)]
            overlap = iou(local[BOX].to_numpy(float), l[l.owner.eq(rid)][BOX].to_numpy(float))
            support[local.index] = overlap.max(axis=1, initial=0)
            clean[rid] = owned[owned.owner.eq(rid)][BOX].to_numpy(float)
    return support, clean


def extra_item(reordered, threshold_alt, clean_boxes, count):
    # Deduplicate all counted evidence before testing for an extra object.
    evidence = list(clean_boxes)
    for _, candidate in reordered[reordered["rank"].ge(threshold_alt)].iterrows():
        xy = candidate[BOX].to_numpy(float)
        if iou(xy[None], np.array(evidence).reshape(-1, 4)).max(initial=0) < .30:
            evidence.append(xy)
    return len(evidence) >= 2 and len(evidence) > count and bool(
        reordered["rank"].ge(threshold_alt).any())
def feature_matrix(pool, vectors, *extra):
    return np.column_stack([vectors, pool.score, pool.name.eq("Book"), *extra])


def manual_item_labels(pool, annotations):
    doc = json.loads(annotations.read_text(encoding="utf-8"))
    records = {record["source_row_id"]: record["objects"] for record in doc["records"].values()}
    labels = np.full(len(pool), -1)
    for rid, group in pool.groupby("owner"):
        truth = np.array([obj["bbox"] for obj in records[rid] if obj["class"] == "item"]).reshape(-1, 4)
        coords = group[BOX].to_numpy(float)
        labels[group.index[iou(coords, truth).max(axis=1, initial=0) < .30]] = 0
        for match in pairs(truth, coords, .50).values():
            labels[group.index[match]] = 1
    return labels


class ItemRanker:
    def __init__(self, inputs, heads):
        self.heads = heads
        self.inputs = inputs
        layout = inputs.test_layout
        raw = inputs.boxes("SAMIR/dfine/tables.csv")
        assert set(layout.frame_id) <= set(raw.frame_id)
        self.teacher = raw[raw.name.isin(ITEM_CLASSES)]
        self.test = candidates(self.teacher, layout)
        self.test_vectors = embeddings(self.test[["frame_id", *BOX]], inputs.cache / "d42/embeddings.npz", "cuda")

    def scores(self, name, test_extra=()):
        head, threshold = self.heads.get(name)
        scores = head.predict_proba(feature_matrix(self.test, self.test_vectors, *test_extra))[:, 1]
        return scores, threshold

    def recover(self, base):
        scores, threshold = self.scores('items')
        chosen = base.index.isin(self.test.loc[scores >= threshold, "owner"])
        return base.assign(items=base["items"] + chosen.astype(int))

    def recover_with_student(self, base):
        inputs = self.inputs
        frame_ids = set(inputs.test_layout.frame_id)
        raw = inputs.boxes("SAMIR/student_pred/tables.csv", frame_ids)
        student = raw[raw.name.eq("item")]
        test_support, clean = box_evidence(self.test, self.teacher, student, inputs.test_layout)
        scores, threshold = self.scores('items_supported', (test_support,))
        added = pd.Series(False, index=base.index)
        for rid, group in self.test.assign(rank=scores).groupby("owner"):
            ranked = group.sort_values("rank", ascending=False)
            added.at[rid] = extra_item(ranked, threshold, clean[rid], base.at[rid, "items"])
        return base.assign(items=base["items"] + added.astype(int))


def prepare_item_heads(inputs, annotations):
    raw = inputs.boxes('SAMIR/dfine/tables.csv')
    teacher = raw[raw.name.isin(ITEM_CLASSES)]
    pool = candidates(teacher, inputs.train_layout)
    labels = manual_item_labels(pool, annotations)
    known = labels >= 0
    raw_student = inputs.boxes('SAMIR/student_pred/tables.csv', set(inputs.train_layout.frame_id))
    support, _ = box_evidence(pool, teacher, raw_student[raw_student.name.eq('item')], inputs.train_layout)
    result = {}
    for name, filename, device, extra in (
        ('items', 'd41/embeddings_cuda.npz', 'cuda', ()),
        ('items_supported', 'd41/embeddings.npz', 'cpu', (support,)),
    ):
        vectors = embeddings(pool[['frame_id', *BOX]], inputs.cache / filename, device)
        x = feature_matrix(pool, vectors, *extra)[known]
        head = fit(x, labels[known])
        threshold = gate(head.predict_proba(x)[:, 1], labels[known])
        result[name] = (head, threshold)
    return result
