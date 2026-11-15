"""Convenience entrypoint; install the package first with pip install -e ."""
import sys
from countsruins.cli import main

if __name__ == "__main__":
    raise SystemExit(main(["reproduce-cached", *sys.argv[1:]]))
