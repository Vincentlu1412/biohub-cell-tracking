"""Analyze Biohub GEFF ground-truth graphs.

Examples
--------
python scripts/analyze_gt.py --data-root /kaggle/input
python scripts/analyze_gt.py --data-root /kaggle/input/biohub-cell-tracking-during-development
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from configs.baseline import (  # noqa: E402
    DEFAULT_DATA_ROOT,
    TEST_DIR_NAME,
    TRAIN_DIR_NAME,
    VOXEL_SIZE_ZYX_UM,
)
from src.data_paths import resolve_data_root  # noqa: E402
from src.geff import add_edge_measurements, count_division_parents, load_gt  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze Biohub GEFF GT files.")
    parser.add_argument(
        "--data-root",
        type=Path,
        default=DEFAULT_DATA_ROOT,
        help="Competition data root containing train/ and test/ directories.",
    )
    parser.add_argument(
        "--save-csv",
        type=Path,
        default=None,
        help="Optional path to save per-sample statistics as CSV.",
    )
    return parser.parse_args()


def find_samples(data_root: Path) -> tuple[list[Path], list[Path], list[Path]]:
    data_root = resolve_data_root(data_root)
    train_dir = data_root / TRAIN_DIR_NAME
    test_dir = data_root / TEST_DIR_NAME

    if not train_dir.exists():
        raise FileNotFoundError(f"Missing train directory: {train_dir}")
    if not test_dir.exists():
        raise FileNotFoundError(f"Missing test directory: {test_dir}")

    train_zarrs = sorted(train_dir.glob("*.zarr"))
    train_geffs = sorted(train_dir.glob("*.geff"))
    test_zarrs = sorted(test_dir.glob("*.zarr"))

    return train_zarrs, train_geffs, test_zarrs


def summarize_sample(geff_path: Path) -> tuple[dict, pd.DataFrame]:
    nodes, edges = load_gt(geff_path)
    measured_edges = add_edge_measurements(nodes, edges, VOXEL_SIZE_ZYX_UM)

    sample_distances = measured_edges["distance_um"].to_numpy(dtype=np.float32)

    record = {
        "dataset": geff_path.stem,
        "nodes": int(len(nodes)),
        "edges": int(len(edges)),
        "annotated_frames": int(nodes["t"].nunique()),
        "t_min": int(nodes["t"].min()) if len(nodes) else -1,
        "t_max": int(nodes["t"].max()) if len(nodes) else -1,
        "divisions": count_division_parents(edges),
        "median_disp_um": float(np.median(sample_distances))
        if len(sample_distances)
        else np.nan,
        "max_disp_um": float(np.max(sample_distances))
        if len(sample_distances)
        else np.nan,
    }

    return record, measured_edges


def print_global_stats(
    train_zarrs: list[Path],
    train_geffs: list[Path],
    test_zarrs: list[Path],
    stats_df: pd.DataFrame,
    all_edges: pd.DataFrame,
) -> None:
    all_distances = all_edges["distance_um"].to_numpy(dtype=np.float32)

    print("=== Dataset files ===")
    print("Train Zarr:", len(train_zarrs))
    print("Train GEFF:", len(train_geffs))
    print("Test Zarr :", len(test_zarrs))

    print("\n=== Dataset statistics ===")
    print("Datasets:", len(stats_df))
    print("Total nodes:", int(stats_df["nodes"].sum()))
    print("Total edges:", int(stats_df["edges"].sum()))
    print("Total divisions:", int(stats_df["divisions"].sum()))

    print("\nNodes / sample:")
    print(stats_df["nodes"].describe())

    print("\nEdges / sample:")
    print(stats_df["edges"].describe())

    print("\nAnnotated frames / sample:")
    print(stats_df["annotated_frames"].describe())

    print("\nDelta t distribution:")
    print(all_edges["dt"].value_counts().sort_index())

    print("\n=== Global displacement (um) ===")
    for q in [50, 75, 90, 95, 97, 99, 99.5, 100]:
        print(f"p{q}: {np.percentile(all_distances, q)}")

    print("\n=== Edges above candidate gates ===")
    for threshold in [5, 7, 8, 9, 10, 12, 15, 20]:
        n_edges = int(np.sum(all_distances > threshold))
        pct = n_edges / len(all_distances) * 100 if len(all_distances) else 0.0
        print(f"> {threshold:2d} um : {n_edges:5d} edges ({pct:.4f}%)")

    division_df = stats_df[stats_df["divisions"] > 0].sort_values(
        "divisions", ascending=False
    )
    print("\n=== Samples with divisions ===")
    if len(division_df) == 0:
        print("None")
    else:
        print(division_df[["dataset", "nodes", "edges", "divisions"]].head(30))


def main() -> None:
    args = parse_args()
    data_root = args.data_root

    train_zarrs, train_geffs, test_zarrs = find_samples(data_root)

    if len(train_geffs) == 0:
        raise FileNotFoundError(f"No *.geff files found under {data_root / TRAIN_DIR_NAME}")

    records = []
    measured_edge_frames = []

    for geff_path in tqdm(train_geffs, desc="Analyzing GEFF"):
        record, measured_edges = summarize_sample(geff_path)
        records.append(record)
        measured_edge_frames.append(measured_edges.assign(dataset=geff_path.stem))

    stats_df = pd.DataFrame.from_records(records).sort_values("dataset").reset_index(drop=True)
    all_edges = pd.concat(measured_edge_frames, ignore_index=True)

    print_global_stats(train_zarrs, train_geffs, test_zarrs, stats_df, all_edges)

    if args.save_csv is not None:
        args.save_csv.parent.mkdir(parents=True, exist_ok=True)
        stats_df.to_csv(args.save_csv, index=False)
        print(f"\nSaved per-sample CSV: {args.save_csv}")


if __name__ == "__main__":
    main()
