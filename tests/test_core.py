import csv
import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from countsruins.assets import check_assets
from countsruins.evaluation.submission import FIELDS, occupancy_ap, score, validate
from countsruins.reproduction.cached import reproduce


class CoreTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="countsruins_tests_")
        self.root = Path(self.temporary.name)

    def tearDown(self):
        self.temporary.cleanup()

    def write_csv(self, name, rows, fields=FIELDS):
        path = self.root / name
        with path.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
        return path

    def row(self, row_id="example", **changes):
        return dict(row_id=row_id, people=0, hands=0, cups=0, dishes=0, items=0,
                    occ_prob=.2) | changes

    def test_fractional_counts_and_id_order(self):
        path = self.write_csv("pred.csv", [self.row("b", cups=.5), self.row("a")])
        self.assertEqual(set(validate(path, ["a", "b"])), {"a", "b"})

    def test_duplicate_ids_rejected(self):
        path = self.write_csv("pred.csv", [self.row(), self.row()])
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            validate(path, ["example"])

    def test_missing_and_extra_rows_rejected(self):
        path = self.write_csv("pred.csv", [self.row("extra")])
        with self.assertRaisesRegex(ValueError, "1 missing, 1 extra"):
            validate(path, ["expected"])

    def test_invalid_numbers_rejected(self):
        for change in ({"cups": "nan"}, {"hands": -1}, {"items": 51}, {"occ_prob": 1.1}):
            with self.subTest(change=change):
                path = self.write_csv("pred.csv", [self.row(**change)])
                with self.assertRaises(ValueError):
                    validate(path, ["example"])

    def test_ap_ties_and_extremes(self):
        self.assertEqual(occupancy_ap([True, False], [.5, .5]), .5)
        self.assertEqual(occupancy_ap([True, True], [.1, .9]), 1.)
        self.assertAlmostEqual(occupancy_ap([False, False], [.1, .3]), .8)

    def test_zero_support_and_ignored_rows(self):
        solution = self.write_csv("labels.csv", [self.row("empty") | {"Usage": "Public"},
                                                 self.row("ignored", cups=8) | {"Usage": "Ignored"}],
                                  (*FIELDS, "Usage"))
        prediction = self.write_csv("pred.csv", [self.row("empty", cups=.5, occ_prob=.2),
                                                 self.row("ignored", cups=0)])
        result = score(solution, prediction)
        self.assertEqual(result["rows"], 1)
        self.assertAlmostEqual(result["score"], .88)

    def test_occupied_consistency(self):
        solution = self.write_csv("labels.csv", [self.row(people=1) | {"occupied": 0}],
                                  (*FIELDS, "occupied"))
        prediction = self.write_csv("pred.csv", [self.row(people=1)])
        with self.assertRaisesRegex(ValueError, "occupied"):
            score(solution, prediction)

    def test_asset_hash_and_missing_file(self):
        path = self.root / "model.bin"
        path.write_bytes(b"synthetic artifact")
        manifest = self.root / "assets.json"
        manifest.write_text(json.dumps({"model.bin": {"bytes": path.stat().st_size,
                             "sha256": hashlib.sha256(path.read_bytes()).hexdigest()},
                             "missing.bin": {"sha256": "0" * 64}}))
        result = check_assets(self.root, manifest)
        self.assertEqual([r["status"] for r in result], ["ok", "missing"])

    def test_asset_path_traversal_rejected(self):
        manifest = self.root / "assets.json"
        manifest.write_text(json.dumps({"../outside": {"sha256": "0" * 64}}))
        with self.assertRaisesRegex(ValueError, "escapes"):
            check_assets(self.root, manifest)

    def test_package_does_not_import_ml_frameworks(self):
        result = subprocess.run([sys.executable, "-c",
            "import countsruins.cli,sys; assert 'torch' not in sys.modules; assert 'transformers' not in sys.modules"],
            capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_reference_wrapper_on_synthetic_repository(self):
        # This tests orchestration, not the actual external D53 reconstruction.
        repo = self.root / "legacy"
        repo.mkdir()
        subprocess.run(["git", "init", "-q", str(repo)], check=True)
        entry = repo / "final"
        entry.mkdir()
        content = "row_id,people,hands,cups,dishes,items,occ_prob\nexample,0,0,0,0,0,0.2\n"
        (entry / "reproduce.py").write_text(
            "import sys,pathlib\n"
            "p=pathlib.Path(sys.argv[sys.argv.index('--work')+1])/'d53'\n"
            "p.mkdir(parents=True)\n"
            f"(p/'submission_d53.csv').write_bytes({content.encode()!r})\n")
        subprocess.run(["git", "-C", str(repo), "add", "."], check=True)
        subprocess.run(["git", "-C", str(repo), "-c", "user.name=Synthetic test",
                        "-c", "user.email=test@example.invalid", "commit", "-qm", "fixture"], check=True)
        commit = subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
        manifests = self.root / "manifests"
        manifests.mkdir()
        (manifests / "source.json").write_text(json.dumps({"commit": commit,
            "entrypoint": "final/reproduce.py", "final_csv": "d53/submission_d53.csv", "expected_rows": 1}))
        bundle = self.root / "bundle.zip"
        bundle.write_bytes(b"synthetic bundle")
        (manifests / "reference.json").write_text(json.dumps({
            "bundle_sha256": hashlib.sha256(bundle.read_bytes()).hexdigest(),
            "stages": {"d53/submission_d53.csv": hashlib.sha256(content.encode()).hexdigest()}}))
        data = self.root / "data"
        data.mkdir()
        (data / "sample_submission.csv").write_text(content)
        work = self.root / "run"
        checked = reproduce(repo, data, bundle, work, manifests, dry_run=True)
        self.assertEqual(checked["status"], "inputs_checked_only")
        self.assertFalse(work.exists())
        output = reproduce(repo, data, bundle, work, manifests)
        self.assertEqual(output["status"], "reproduced")
        with self.assertRaisesRegex(ValueError, "new --work"):
            reproduce(repo, data, bundle, work, manifests)


if __name__ == "__main__":
    unittest.main()
