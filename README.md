# Biohub Cell Tracking Baseline

First-version local GitHub project for the Biohub Cell Tracking competition.

## Current stage

This initial commit contains the data/GT analysis layer:

- `requirements.txt`
- `configs/baseline.py`
- `src/geff.py`
- `scripts/analyze_gt.py`
- `src/preprocessing.py`
- `src/targets.py`
- `src/dataset.py`
- `src/model.py`
- `src/losses.py`
- `scripts/train_detector.py`

The first target is to reproduce the global GT statistics from the Kaggle training set before building the detector and tracker.

## Run

```bash
pip install -r requirements.txt
python scripts/analyze_gt.py --data-root /kaggle/input
```

For local development, replace `--data-root` with the directory containing `train/` and `test/`.

Expected headline output from the current training data:

```text
Train Zarr: 199
Train GEFF: 199
Test Zarr : 4
Total nodes: 133318
Total edges: 128883
Total divisions: 151
Delta t distribution: only 1-frame edges
p50 ~= 1.8168 um
p95 ~= 5.3434 um
p99 ~= 8.3849 um
p99.5 ~= 9.4229 um
```

## Train Detector

Start with a short smoke run:

```bash
python scripts/train_detector.py \
    --data-root /kaggle/input \
    --epochs 1 \
    --samples-per-epoch 64 \
    --batch-size 1 \
    --num-workers 2
```

Then increase `--samples-per-epoch` and `--epochs` for a real run.
