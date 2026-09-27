"""Generate a Biohub submission CSV from test images."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
from tqdm import tqdm
import zarr


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from configs.baseline import DEFAULT_DATA_ROOT, IMAGE_ARRAY_KEY  # noqa: E402
from src.data_paths import resolve_data_root  # noqa: E402
from src.detection import assign_node_ids, detect_centers_heuristic  # noqa: E402
from src.submission import sample_to_submission_rows, validate_submission, write_submission  # noqa: E402
from src.tracking import link_detections  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate Biohub submission.csv.")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output", type=Path, default=Path("/kaggle/working/submission.csv"))
    parser.add_argument("--percentile", type=float, default=99.7)
    parser.add_argument("--max-detections-per-frame", type=int, default=512)
    parser.add_argument("--max-frames", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_root = resolve_data_root(args.data_root)
    test_dir = data_root / "test"
    test_zarrs = sorted(test_dir.glob("*.zarr"))

    if not test_zarrs:
        raise FileNotFoundError(f"No test zarrs found under {test_dir}")

    all_rows: list[dict] = []

    for zarr_path in test_zarrs:
        dataset = zarr_path.stem
        print(f"Predicting {dataset}")

        image = zarr.open_group(str(zarr_path), mode="r")[IMAGE_ARRAY_KEY]
        n_frames = image.shape[0] if args.max_frames is None else min(args.max_frames, image.shape[0])

        frame_detections = []
        for t in tqdm(range(n_frames), desc=dataset):
            detections = detect_centers_heuristic(
                image[t],
                frame=t,
                percentile=args.percentile,
                max_detections=args.max_detections_per_frame,
            )
            frame_detections.append(detections)

        nodes = pd.concat(frame_detections, ignore_index=True)
        nodes = assign_node_ids(nodes, start_id=1)
        edges = link_detections(nodes)

        print(f"{dataset}: nodes={len(nodes)} edges={len(edges)}")
        all_rows.extend(sample_to_submission_rows(dataset, nodes, edges))

    submission = write_submission(all_rows, args.output)
    validate_submission(submission)

    print(f"Wrote {args.output}")
    print(f"Rows: {len(submission)}")
    print(submission["row_type"].value_counts())


if __name__ == "__main__":
    main()
