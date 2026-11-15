"""Load organizer metadata and validate frozen detector outputs and embeddings."""
from hashlib import sha256
from importlib.util import module_from_spec, spec_from_file_location
import json
from pathlib import Path
import shutil
from zipfile import ZipFile

import numpy as np
import pandas as pd

ASSETS = Path(__file__).resolve().parents[3] / "manifests"
MODEL = "facebook/dinov2-small"
REVISION = "ed25f3a31f01632728cabb09d1542f84ab7b0056"
WEIGHT_HASH = "ae1e99fcefd534ed978cdeb8326f08030c96e28b7a81ffcbc98a857c84d14be1"


INPUTS = (
    "SAMIR/submission_student_nb_pure.csv",
    "SAMIR/submission_dfine_clean.csv",
    "night-followup/submission_night_best.csv",
    "submission-exp016/submission_exp016.csv",
    "submission-exp016/yoloe/detections.jsonl",
    "SAMIR/dfine/tables.csv", "SAMIR/dfine/context.csv",
    "SAMIR/student_pred/tables.csv", "SAMIR/student_pred/context.csv",
    "d6/instances.csv", "d7/embeddings.npz",
    "d13/annotations.json", "d13/cases.json",
    "d21/embeddings.npz", "d41/embeddings.npz", "d42/embeddings.npz", "d41/embeddings_cuda.npz",
)


def digest(path):
    checksum = sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024*1024), b""):
            checksum.update(block)
    return checksum.hexdigest()


def unpack(bundle, work):
    with ZipFile(bundle) as package:
        manifest = json.loads(package.read("manifest.json"))
        if set(manifest) != set(INPUTS):
            raise ValueError("Unexpected input manifest")
        for name, expected in manifest.items():
            destination = work/name
            if destination.exists():
                raise FileExistsError(destination)
            destination.parent.mkdir(parents=True, exist_ok=True)
            with package.open(name) as src, destination.open("wb") as dst:
                shutil.copyfileobj(src, dst)
            if digest(destination) != expected:
                raise ValueError(f"Input hash mismatch: {name}")


class Inputs:
    def __init__(self, data, cache, training=False):
        self.data = data
        self.cache = cache
        self.sample = pd.read_csv(data / "sample_submission.csv")
        self.tables = pd.read_csv(data / "tables.csv")
        if training:
            self.labels = pd.read_csv(data / 'train_labels.csv').set_index('row_id')
            self.train_layout = self.tables[self.tables.row_id.isin(self.labels.index)].reset_index(drop=True)
        self.test_layout = self.tables.set_index("row_id").loc[self.sample.row_id].reset_index()
        self.frames = pd.read_csv(data / "frames.csv").set_index("frame_id")
        self.targets = pd.read_csv(data / "targets.csv").fillna("").set_index("frame_id")
        self.production = pd.read_csv(data / "detections.csv")

        spec = spec_from_file_location("organizer_score", data / "score.py")
        self.scorer = module_from_spec(spec)
        spec.loader.exec_module(self.scorer)

    def boxes(self, filename, frames=None):
        if frames is None:
            return pd.read_csv(self.cache / filename)
        chunks = pd.read_csv(self.cache / filename, chunksize=100_000)
        return pd.concat([part[part.frame_id.isin(frames)] for part in chunks], ignore_index=True)

    def grouped_boxes(self, filename, frames, names):
        detections = self.boxes(filename, frames)
        return dict(tuple(detections[detections.name.isin(names)].groupby("frame_id")))

    def predictions(self, filename):
        csv = pd.read_csv(self.cache / filename, float_precision="round_trip")
        return self.scorer.validate_submission(csv, self.sample.row_id).set_index("row_id")


def embeddings(rows, cache, device, metadata=None):
    if metadata is None:
        metadata = {
            "model": MODEL, "revision": REVISION, "license": "Apache-2.0",
            "source": f"https://huggingface.co/{MODEL}/tree/{REVISION}",
            "sha256": WEIGHT_HASH, "image_size": 112, "padding": .20,
            "crops": len(rows), "device": device,
        }
    signature = sha256(rows.to_csv(index=False, lineterminator="\r\n").encode() + json.dumps(metadata).encode()).hexdigest()
    with np.load(cache) as saved:
        if str(saved["signature"]) != signature:
            raise ValueError(f"Embedding cache signature mismatch: {cache}")
        vectors = saved["features"]
    if vectors.shape != (len(rows), 384) or not np.isfinite(vectors).all():
        raise ValueError(f"Invalid cached embeddings: {cache}")
    return vectors
