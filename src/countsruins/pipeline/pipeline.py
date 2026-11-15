"""Assemble the final submission and verify reproducibility."""
import json

import numpy as np
import pandas as pd
from scipy.special import expit, logit

from .cups import reconcile_cups, recover_cups
from .inputs import ASSETS, Inputs, digest, unpack
from .items import ItemRanker
from .heads import Heads

COUNTS = ["people", "hands", "cups", "dishes", "items"]


def baseline(inputs):
    student = inputs.predictions("SAMIR/submission_student_nb_pure.csv")
    clean = inputs.predictions("SAMIR/submission_dfine_clean.csv")["items"]
    blend = inputs.predictions("night-followup/submission_night_best.csv")["items"]
    ridge = inputs.predictions("submission-exp016/submission_exp016.csv")["items"]
    np.testing.assert_allclose(blend, .25 * ridge + .75 * clean, rtol=0, atol=1e-12)
    gap = (clean - blend).clip(lower=0)
    return student.assign(hands=np.minimum(student.hands, 2 * student.people),
                          items=blend + .50 * gap.where(clean.ge(2), 0))


def integer_counts(prediction):
    result = prediction.copy()
    result[COUNTS] = np.floor(result[COUNTS] + .5).astype(int)
    return result


def adjust_occupancy(prediction):
    prior = logit(prediction.occ_prob.clip(1e-6, 1 - 1e-6))
    occupied = prediction.people.gt(0).astype(int)
    return prediction.assign(occ_prob=expit(prior + np.log(2) * (2 * occupied - 1)))


def save_stage(prediction, filename, inputs, expected):
    csv = prediction.reset_index()[inputs.sample.columns]
    inputs.scorer.validate_submission(csv, inputs.sample.row_id)
    csv.to_csv(filename, index=False, lineterminator="\r\n")
    checksum = digest(filename)
    if checksum != expected:
        raise ValueError(f"SHA-256 mismatch at {filename.name}: expected {expected}, got {checksum}")
    print(f"{filename.name}  {checksum}", flush=True)
    return pd.read_csv(filename, float_precision="round_trip").set_index("row_id")


def reproduce(data, bundle, work, heads_path):
    heads = Heads.load(heads_path)
    reference = json.loads((ASSETS / "reference_compact.json").read_text(encoding="utf-8"))
    if digest(bundle) != reference["bundle_sha256"]:
        raise ValueError("Unexpected input bundle SHA-256")
    work.mkdir(parents=True, exist_ok=False)
    cache = work / "cache"
    unpack(bundle, cache)
    inputs = Inputs(data, cache)
    if len(inputs.sample) != 4733 or not inputs.sample.row_id.is_unique:
        raise ValueError("Expected 4733 unique rows in the organizer submission template")
    stages = work / "stages"
    stages.mkdir()
    hashes = reference["stages"]

    cups = save_stage(recover_cups(inputs, baseline(inputs), heads), stages / "cups.csv", inputs, hashes["cups"])
    rounded = save_stage(integer_counts(cups), stages / "rounded.csv", inputs, hashes["rounded"])
    ranker = ItemRanker(inputs, heads)
    items = save_stage(ranker.recover(rounded), stages / "items.csv", inputs, hashes["items"])
    supported = save_stage(ranker.recover_with_student(items), stages / "supported_items.csv",
                           inputs, hashes["supported_items"])
    final = adjust_occupancy(reconcile_cups(inputs, supported))
    save_stage(final, work / "submission.csv", inputs, hashes["final"])
    print(f"Submission: {work / 'submission.csv'}", flush=True)
