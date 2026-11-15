from pathlib import Path
import argparse
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent / 'src'))

from countsruins.pipeline.pipeline import reproduce

if __name__ == '__main__':
    p = argparse.ArgumentParser(description='Predict table counts using saved models and features')
    p.add_argument('--data', type=Path, required=True)
    p.add_argument('--bundle', type=Path, default=Path('artifacts/bundles/final_d53_cpu_inputs.zip'))
    p.add_argument('--models', type=Path, default=Path(__file__).resolve().parent / 'models/heads.json')
    p.add_argument('--work', type=Path, default=Path('artifacts/result'))
    args = p.parse_args()
    reproduce(args.data.resolve(), args.bundle.resolve(), args.work.resolve(), args.models.resolve())
