"""Utilities for reading Biohub GEFF ground-truth files.

The competition GEFF files are Zarr groups with this layout:

    nodes/ids
    nodes/props/t/values
    nodes/props/z/values
    nodes/props/y/values
    nodes/props/x/values
    edges/ids

This module converts that graph into small pandas DataFrames used by analysis,
training target generation, and tracking validation.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import zarr


NODE_COLUMNS = ["node_id", "t", "z", "y", "x"]
EDGE_COLUMNS = ["source_id", "target_id"]


@dataclass(frozen=True)
class GeffGraph:
    """Ground-truth graph loaded from one GEFF file."""

    nodes: pd.DataFrame
    edges: pd.DataFrame


def _require_array(group: zarr.Group, path: str) -> zarr.Array:
    """Return a required array and raise a useful error when it is missing."""

    try:
        obj = group[path]
    except KeyError as exc:
        raise KeyError(f"Missing required GEFF array: {path}") from exc

    if not isinstance(obj, zarr.Array):
        raise TypeError(f"Expected GEFF path {path!r} to be a zarr.Array")

    return obj


def load_geff(geff_path: str | Path) -> GeffGraph:
    """Load nodes and edges from one Biohub GEFF file.

    Parameters
    ----------
    geff_path:
        Path to a ``*.geff`` Zarr group.

    Returns
    -------
    GeffGraph
        ``nodes`` has columns ``node_id, t, z, y, x``.
        ``edges`` has columns ``source_id, target_id``.
    """

    geff_path = Path(geff_path)
    group = zarr.open_group(str(geff_path), mode="r")

    node_ids = _require_array(group, "nodes/ids")[:].astype(np.int64)
    t = _require_array(group, "nodes/props/t/values")[:].astype(np.int64)
    z = _require_array(group, "nodes/props/z/values")[:].astype(np.int64)
    y = _require_array(group, "nodes/props/y/values")[:].astype(np.int64)
    x = _require_array(group, "nodes/props/x/values")[:].astype(np.int64)

    nodes = pd.DataFrame(
        {
            "node_id": node_ids,
            "t": t,
            "z": z,
            "y": y,
            "x": x,
        },
        columns=NODE_COLUMNS,
    )

    edge_ids = _require_array(group, "edges/ids")[:].astype(np.int64)
    if edge_ids.ndim != 2 or edge_ids.shape[1] != 2:
        raise ValueError(
            f"Expected edges/ids to have shape (n_edges, 2), got {edge_ids.shape}"
        )

    edges = pd.DataFrame(
        {
            "source_id": edge_ids[:, 0],
            "target_id": edge_ids[:, 1],
        },
        columns=EDGE_COLUMNS,
    )

    return GeffGraph(nodes=nodes, edges=edges)


def load_gt(geff_path: str | Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compatibility wrapper returning ``(nodes, edges)``."""

    graph = load_geff(geff_path)
    return graph.nodes, graph.edges


def validate_graph(nodes: pd.DataFrame, edges: pd.DataFrame) -> None:
    """Run cheap consistency checks for one loaded graph."""

    missing_node_cols = sorted(set(NODE_COLUMNS) - set(nodes.columns))
    missing_edge_cols = sorted(set(EDGE_COLUMNS) - set(edges.columns))

    if missing_node_cols:
        raise ValueError(f"nodes is missing columns: {missing_node_cols}")
    if missing_edge_cols:
        raise ValueError(f"edges is missing columns: {missing_edge_cols}")

    node_ids = set(nodes["node_id"].astype(np.int64).tolist())
    edge_ids: Iterable[int] = pd.concat(
        [edges["source_id"], edges["target_id"]], ignore_index=True
    ).astype(np.int64)

    unknown_ids = sorted(set(edge_ids) - node_ids)
    if unknown_ids:
        preview = unknown_ids[:10]
        raise ValueError(f"edges reference unknown node ids, first examples: {preview}")


def add_edge_measurements(
    nodes: pd.DataFrame,
    edges: pd.DataFrame,
    voxel_size_zyx_um: tuple[float, float, float],
) -> pd.DataFrame:
    """Attach dt and physical displacement to every edge.

    The returned DataFrame keeps ``source_id`` and ``target_id`` and adds:
    ``source_t``, ``target_t``, ``dt``, and ``distance_um``.
    """

    validate_graph(nodes, edges)

    node_lookup = nodes.set_index("node_id", drop=False)
    voxel_size = np.asarray(voxel_size_zyx_um, dtype=np.float32)

    records = []
    for source_id, target_id in edges[["source_id", "target_id"]].itertuples(index=False):
        source = node_lookup.loc[int(source_id)]
        target = node_lookup.loc[int(target_id)]

        source_zyx = source[["z", "y", "x"]].to_numpy(dtype=np.float32)
        target_zyx = target[["z", "y", "x"]].to_numpy(dtype=np.float32)
        distance_um = float(np.linalg.norm((target_zyx - source_zyx) * voxel_size))

        records.append(
            {
                "source_id": int(source_id),
                "target_id": int(target_id),
                "source_t": int(source["t"]),
                "target_t": int(target["t"]),
                "dt": int(target["t"] - source["t"]),
                "distance_um": distance_um,
            }
        )

    return pd.DataFrame.from_records(
        records,
        columns=[
            "source_id",
            "target_id",
            "source_t",
            "target_t",
            "dt",
            "distance_um",
        ],
    )


def count_division_parents(edges: pd.DataFrame) -> int:
    """Count source nodes with two or more outgoing GT edges."""

    if len(edges) == 0:
        return 0

    out_degree = edges.groupby("source_id", sort=False).size()
    return int((out_degree >= 2).sum())
