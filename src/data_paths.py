"""Helpers for locating Kaggle-mounted Biohub data."""

from __future__ import annotations

from pathlib import Path


def resolve_data_root(data_root: str | Path) -> Path:
    """Return the directory that directly contains train/ and test/."""

    data_root = Path(data_root).expanduser().resolve()

    if _looks_like_data_root(data_root):
        return data_root

    candidates: list[tuple[int, Path]] = []
    for train_dir in data_root.rglob("train"):
        if not train_dir.is_dir():
            continue

        candidate = train_dir.parent
        if _looks_like_data_root(candidate):
            geff_count = len(list(train_dir.glob("*.geff")))
            candidates.append((geff_count, candidate))

    if candidates:
        candidates.sort(key=lambda item: item[0], reverse=True)
        resolved = candidates[0][1]
        print(f"Resolved data root: {resolved}")
        return resolved

    raise FileNotFoundError(
        "Could not find Biohub data root. Expected a directory containing "
        "train/*.geff, train/*.zarr, and test/*.zarr. "
        f"Received: {data_root}"
    )


def _looks_like_data_root(path: Path) -> bool:
    train_dir = path / "train"
    test_dir = path / "test"

    return (
        train_dir.is_dir()
        and test_dir.is_dir()
        and any(train_dir.glob("*.geff"))
        and any(train_dir.glob("*.zarr"))
        and any(test_dir.glob("*.zarr"))
    )
