import json
import os
import subprocess
import sys
from pathlib import Path

from ..assets import sha256
from ..evaluation.submission import read_rows, unique_ids, validate


def reproduce(legacy_repo: Path, data: Path, bundle: Path, work: Path,
              manifests: Path, dry_run=False):
    source = json.loads((manifests / "source.json").read_text(encoding="utf-8"))
    reference = json.loads((manifests / "reference.json").read_text(encoding="utf-8"))
    legacy_repo, data, bundle, work = [path.resolve() for path in (legacy_repo, data, bundle, work)]
    if not (data / "sample_submission.csv").is_file():
        raise ValueError("--data must contain sample_submission.csv from the organizer package")
    if work.exists():
        raise ValueError("Use a new --work directory; existing files are never overwritten")
    commit = subprocess.run(["git", "-C", str(legacy_repo), "rev-parse", "HEAD"],
                            capture_output=True, text=True, check=True).stdout.strip()
    if commit != source["commit"]:
        raise ValueError(f"Legacy checkout must be at {source['commit']}, got {commit}")
    for command in (["git", "-C", str(legacy_repo), "diff", "--quiet"],
                    ["git", "-C", str(legacy_repo), "diff", "--cached", "--quiet"]):
        if subprocess.run(command, check=False).returncode:
            raise ValueError("Legacy checkout has modified tracked files")
    entrypoint = legacy_repo / source["entrypoint"]
    if not entrypoint.is_file():
        raise ValueError(f"Missing reference entrypoint: {entrypoint}")
    if not bundle.is_file():
        raise ValueError(f"External bundle not found: {bundle}")
    if sha256(bundle) != reference["bundle_sha256"]:
        raise ValueError("CPU bundle SHA-256 does not match the reference")
    command = [sys.executable, "-B", str(entrypoint), "reproduce", "--data", str(data),
               "--work", str(work), "--bundle", str(bundle)]
    if dry_run:
        return {"status": "inputs_checked_only", "command": command,
                "note": "No models, heads or CSV were executed/generated"}
    environment = os.environ | {"D53_FROZEN_ENCODER": "1", "HF_HUB_OFFLINE": "1",
                                "CUDA_VISIBLE_DEVICES": ""}
    subprocess.run(command, cwd=legacy_repo, env=environment, check=True)
    output = work / source["final_csv"]
    if sha256(output) != reference["stages"][source["final_csv"]]:
        raise ValueError("Final CSV SHA-256 mismatch")
    _, template = read_rows(data / "sample_submission.csv")
    values = validate(output, unique_ids(template))
    if len(values) != source["expected_rows"]:
        raise ValueError("Unexpected final row count")
    return {"status": "reproduced", "submission": str(output), "rows": len(values),
            "sha256": sha256(output)}
