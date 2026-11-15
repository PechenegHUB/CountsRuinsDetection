"""Small binary classifiers stored as plain numerical parameters."""
import json
from pathlib import Path

import numpy as np
from scipy.special import expit

NAMES = {'cups': 384, 'items': 386, 'items_supported': 387}


class LinearHead:
    def __init__(self, coefficients, intercept):
        self.coef = np.asarray(coefficients, dtype=np.float64)
        self.intercept = np.asarray(intercept, dtype=np.float64)
        if self.coef.ndim != 2 or self.coef.shape[0] != 1 or self.intercept.shape != (1,):
            raise ValueError('Expected a binary linear classifier')
        if not np.isfinite(self.coef).all() or not np.isfinite(self.intercept).all():
            raise ValueError('Nonfinite classifier parameters')

    def predict_proba(self, x):
        x = np.asarray(x)
        if x.ndim != 2 or x.shape[1] != self.coef.shape[1] or not np.isfinite(x).all():
            raise ValueError('Invalid classifier features')
        probability = expit((x @ self.coef.T + self.intercept).ravel())
        return np.vstack([1 - probability, probability]).T


class Heads:
    def __init__(self, document):
        if document.get('format') != 1 or set(document['heads']) != set(NAMES):
            raise ValueError('Unsupported heads file')
        self.document = document
        self.models = {}
        for name, size in NAMES.items():
            entry = document['heads'][name]
            model = LinearHead(entry['coef'], entry['intercept'])
            threshold = float(entry['threshold'])
            if model.coef.shape[1] != size or not np.isfinite(threshold) or not 0 <= threshold <= 1:
                raise ValueError('Invalid head dimensions or threshold: ' + name)
            self.models[name] = (model, threshold)

    def get(self, name):
        return self.models[name]

    @classmethod
    def load(cls, path):
        return cls(json.loads(Path(path).read_text(encoding='utf-8')))


def export(models, path, provenance):
    entries = {}
    for name, (model, threshold) in models.items():
        if list(model.classes_) != [0, 1]:
            raise ValueError('Unexpected classifier class order')
        entries[name] = {'coef': model.coef_.tolist(), 'intercept': model.intercept_.tolist(),
                         'threshold': float(threshold)}
    doc = {'format': 1, 'provenance': provenance, 'heads': entries}
    Heads(doc)
    path = Path(path)
    if path.exists():
        raise FileExistsError('Choose a new heads output file')
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc, indent=2) + '\n', encoding='utf-8')
