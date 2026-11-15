import argparse
import json
import subprocess
import sys
from pathlib import Path

from .assets import check_assets
from .evaluation.submission import read_rows, score, unique_ids, validate


def main(argv=None):
    parser = argparse.ArgumentParser(description="D53 asset checks, CSV validation and cached-reference launcher")
    commands = parser.add_subparsers(dest="command", required=True)
    assets = commands.add_parser("check-assets", help="Check externally supplied files by SHA-256")
    assets.add_argument("--root", required=True, type=Path)
    assets.add_argument("--manifest", required=True, type=Path)
    assets.add_argument("--only", action="append", help="Check one named manifest entry; repeat if needed")
    validation = commands.add_parser("validate", help="Validate a submission against sample row IDs")
    validation.add_argument("submission", type=Path)
    validation.add_argument("--sample", required=True, type=Path)
    scoring = commands.add_parser("score", help="Score predictions using count labels")
    scoring.add_argument("solution", type=Path)
    scoring.add_argument("submission", type=Path)
    cached = commands.add_parser("reproduce-cached", help="Delegate D53 reconstruction to a pinned upstream checkout")
    for name in ("legacy-repo", "data", "bundle", "work"):
        cached.add_argument("--" + name, required=True, type=Path)
    cached.add_argument("--manifests", type=Path, default=Path("manifests"))
    cached.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    try:
        status = 0
        if args.command == "check-assets":
            result = check_assets(args.root, args.manifest, args.only)
            status = int(any(row["status"] != "ok" for row in result))
        elif args.command == "validate":
            _, template = read_rows(args.sample)
            result = {"valid": True, "rows": len(validate(args.submission, unique_ids(template)))}
        elif args.command == "score":
            result = score(args.solution, args.submission)
        else:
            from .reproduction.cached import reproduce
            result = reproduce(args.legacy_repo, args.data, args.bundle, args.work,
                               args.manifests, args.dry_run)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return status
    except (ValueError, OSError, subprocess.CalledProcessError, json.JSONDecodeError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2
