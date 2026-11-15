"""Streaming checksums for externally supplied artifacts; no automatic download."""
import hashlib
import json
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def check_assets(root: Path, manifest_path: Path, only=None):
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    names = list(manifest) if not only else only
    unknown = set(names) - set(manifest)
    if unknown:
        raise ValueError(f"Unknown manifest entries: {sorted(unknown)}")
    resolved_root = root.resolve()
    results = []
    for name in names:
        target = (root / name).resolve()
        if not target.is_relative_to(resolved_root):
            raise ValueError(f"Manifest entry escapes artifact directory: {name}")
        expected = manifest[name]
        status = "missing"
        if target.is_file():
            if expected.get("bytes") is not None and target.stat().st_size != expected["bytes"]:
                status = "size_mismatch"
            else:
                status = "ok" if sha256(target) == expected["sha256"] else "sha256_mismatch"
        results.append({"file": name, "status": status})
    return results
