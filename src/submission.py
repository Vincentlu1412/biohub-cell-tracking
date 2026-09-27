"""Submission CSV writer."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


SUBMISSION_COLUMNS = [
    "id",
    "dataset",
    "row_type",
    "node_id",
    "t",
    "z",
    "y",
    "x",
    "source_id",
    "target_id",
]


def sample_to_submission_rows(
    dataset: str,
    nodes: pd.DataFrame,
    edges: pd.DataFrame,
) -> list[dict]:
    """Convert one sample prediction to submission rows."""

    rows: list[dict] = []

    for row in nodes.sort_values(["t", "node_id"]).itertuples(index=False):
        rows.append(
            {
                "dataset": dataset,
                "row_type": "node",
                "node_id": int(row.node_id),
                "t": int(row.t),
                "z": int(row.z),
                "y": int(row.y),
                "x": int(row.x),
                "source_id": -1,
                "target_id": -1,
            }
        )

    for row in edges.itertuples(index=False):
        rows.append(
            {
                "dataset": dataset,
                "row_type": "edge",
                "node_id": -1,
                "t": -1,
                "z": -1,
                "y": -1,
                "x": -1,
                "source_id": int(row.source_id),
                "target_id": int(row.target_id),
            }
        )

    return rows


def write_submission(rows: list[dict], output_path: str | Path) -> pd.DataFrame:
    """Write rows to Kaggle submission CSV with consecutive id."""

    df = pd.DataFrame(rows)
    if len(df) == 0:
        df = pd.DataFrame(columns=SUBMISSION_COLUMNS[1:])

    df.insert(0, "id", range(len(df)))
    df = df[SUBMISSION_COLUMNS]

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    return df


def validate_submission(df: pd.DataFrame) -> None:
    """Run basic format checks before writing or submitting."""

    missing = [col for col in SUBMISSION_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"Submission is missing columns: {missing}")

    if not df["id"].tolist() == list(range(len(df))):
        raise ValueError("Submission id column must be consecutive from 0.")

    bad_types = set(df["row_type"].unique()) - {"node", "edge"}
    if bad_types:
        raise ValueError(f"Unexpected row_type values: {sorted(bad_types)}")
