"""Simple nearest-neighbor tracking."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

from configs.baseline import TRACK_GATE_UM, VOXEL_SIZE_ZYX_UM


def link_detections(
    nodes: pd.DataFrame,
    gate_um: float = TRACK_GATE_UM,
    voxel_size_zyx_um: tuple[float, float, float] = VOXEL_SIZE_ZYX_UM,
) -> pd.DataFrame:
    """Link detections between adjacent frames with one-to-one assignment."""

    if len(nodes) == 0:
        return pd.DataFrame(columns=["source_id", "target_id"])

    voxel_size = np.asarray(voxel_size_zyx_um, dtype=np.float32)
    edges: list[dict[str, int]] = []

    frames = sorted(nodes["t"].unique().tolist())
    by_t = {int(t): df for t, df in nodes.groupby("t", sort=False)}

    for t in frames:
        current = by_t.get(int(t))
        nxt = by_t.get(int(t) + 1)
        if current is None or nxt is None or len(current) == 0 or len(nxt) == 0:
            continue

        current_xyz = current[["z", "y", "x"]].to_numpy(dtype=np.float32)
        next_xyz = nxt[["z", "y", "x"]].to_numpy(dtype=np.float32)

        diff = (current_xyz[:, None, :] - next_xyz[None, :, :]) * voxel_size
        dist = np.linalg.norm(diff, axis=2)

        row_ind, col_ind = linear_sum_assignment(dist)
        for row, col in zip(row_ind, col_ind):
            if dist[row, col] <= gate_um:
                edges.append(
                    {
                        "source_id": int(current.iloc[row]["node_id"]),
                        "target_id": int(nxt.iloc[col]["node_id"]),
                    }
                )

    return pd.DataFrame(edges, columns=["source_id", "target_id"])
