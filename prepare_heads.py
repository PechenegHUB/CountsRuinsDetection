from pathlib import Path
import argparse
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent / 'src'))

from countsruins.pipeline.prepare import prepare

if __name__ == '__main__':
    p = argparse.ArgumentParser(description='Prepare verifier models once')
    for name in ['data', 'bundle', 'annotations', 'work', 'output']:
        p.add_argument('--'+name, type=Path, required=True)
    a = p.parse_args()
    prepare(a.data.resolve(), a.bundle.resolve(), a.work.resolve(), a.annotations.resolve(), a.output.resolve())
