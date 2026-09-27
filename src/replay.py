"""Replay recorded feature rows through the saved IDS pipeline."""
import argparse
from pathlib import Path
import time
import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--csv', type=Path, default=ROOT / 'data/demo_features.csv')
    parser.add_argument('--delay', type=float, default=0.3)
    args = parser.parse_args()
    if args.delay < 0:
        parser.error('--delay must be non-negative')
    path = ROOT / 'models/ids_pipeline.joblib'
    if not path.exists():
        parser.error('Train first: python src/train.py')
    # Never load an untrusted pickle/joblib file.
    model = joblib.load(path)
    frame = pd.read_csv(args.csv)
    expected = list(model.feature_names_in_)
    missing = sorted(set(expected) - set(frame.columns))
    if missing:
        parser.error('Missing feature columns: ' + ', '.join(missing))
    print('SIMULATED STREAM — recorded rows; no live packet capture')
    attack_index = list(model.classes_).index(1)
    for i in range(len(frame)):
        start = time.perf_counter()
        scores = model.predict_proba(frame[expected].iloc[[i]])[0]
        prediction = int(model.classes_[scores.argmax()])
        elapsed = (time.perf_counter() - start) * 1000
        print(f'{i+1:03d} | {"ATTACK" if prediction else "NORMAL":6} | attack score {scores[attack_index]:.3f} | {elapsed:.1f} ms', flush=True)
        time.sleep(args.delay)

if __name__ == '__main__':
    main()
