"""Detection helpers for inference."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import ndimage as ndi


def detect_centers_heuristic(
    volume_zyx: np.ndarray,
    frame: int,
    percentile: float = 99.7,
    min_distance_zyx: tuple[int, int, int] = (2, 5, 5),
    max_detections: int = 512,
) -> pd.DataFrame:
    """Detect bright local maxima in one ZYX frame.

    This is a CPU-friendly baseline for producing a valid submission before the
    learned detector is fully calibrated.
    """

    volume = volume_zyx.astype(np.float32, copy=False)
    threshold = float(np.percentile(volume, percentile))

    footprint = tuple(2 * int(v) + 1 for v in min_distance_zyx)
    local_max = volume == ndi.maximum_filter(volume, size=footprint, mode="nearest")
    mask = local_max & (volume >= threshold)

    z, y, x = np.nonzero(mask)
    if len(z) == 0:
        return pd.DataFrame(columns=["t", "z", "y", "x", "score"])

    scores = volume[z, y, x]
    order = np.argsort(scores)[::-1][:max_detections]

    detections = pd.DataFrame(
        {
            "t": int(frame),
            "z": z[order].astype(np.int64),
            "y": y[order].astype(np.int64),
            "x": x[order].astype(np.int64),
            "score": scores[order].astype(np.float32),
        }
    )
    return detections.sort_values(["t", "score"], ascending=[True, False]).reset_index(
        drop=True
    )


def assign_node_ids(detections: pd.DataFrame, start_id: int = 1) -> pd.DataFrame:
    """Assign consecutive node IDs to detection rows."""

    detections = detections.copy().reset_index(drop=True)
    detections.insert(0, "node_id", np.arange(start_id, start_id + len(detections)))
    return detections
