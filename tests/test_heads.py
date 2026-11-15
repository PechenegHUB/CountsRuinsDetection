import json
from pathlib import Path
import tempfile
import unittest

try:
    import numpy as np
    from sklearn.linear_model import LogisticRegression
    from countsruins.pipeline.heads import Heads, export
except ImportError:
    np = None


@unittest.skipIf(np is None, 'Install requirements/cached.txt for model tests')
class SavedHeadTests(unittest.TestCase):
    def test_exported_parameters_preserve_probabilities_and_thresholds(self):
        rng = np.random.default_rng(53)
        models, examples = {}, {}
        for name, width in [('cups', 384), ('items', 386), ('items_supported', 387)]:
            x = rng.normal(size=(20, width))
            y = np.arange(20) % 2
            model = LogisticRegression(random_state=42).fit(x, y)
            models[name] = (model, .6)
            examples[name] = rng.normal(size=(8, width)).astype(np.float32)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'heads.json'
            export(models, path, {'test': True})
            saved = Heads.load(path)
            for name, (original, _) in models.items():
                restored, threshold = saved.get(name)
                np.testing.assert_array_equal(restored.predict_proba(examples[name]),
                                              original.predict_proba(examples[name]))
                self.assertEqual(threshold, .6)
            with self.assertRaises(FileExistsError):
                export(models, path, {})
            broken = json.loads(path.read_text())
            broken['heads']['cups']['coef'][0][0] = float('nan')
            with self.assertRaises(ValueError):
                Heads(broken)
