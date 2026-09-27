"""Small Zarr array reader with a no-zarr fallback.

Kaggle final re-runs can disable internet and may not include the ``zarr``
Python package. This module keeps inference usable by first trying the real
package and then falling back to a minimal reader for simple Zarr v2 directory
arrays.
"""

from __future__ import annotations

import gzip
import json
import zlib
from pathlib import Path
from typing import Any

import numpy as np


class MinimalZarrArray:
    """Read a basic Zarr v2 array from a directory store."""

    def __init__(self, array_dir: str | Path) -> None:
        self.array_dir = Path(array_dir)
        meta_path = self.array_dir / ".zarray"
        if not meta_path.exists():
            raise FileNotFoundError(f"Missing Zarr metadata: {meta_path}")

        self.meta: dict[str, Any] = json.loads(meta_path.read_text())
        self.shape = tuple(int(v) for v in self.meta["shape"])
        self.chunks = tuple(int(v) for v in self.meta["chunks"])
        self.dtype = np.dtype(self.meta["dtype"])
        self.order = str(self.meta.get("order", "C"))
        self.fill_value = self.meta.get("fill_value", 0)
        self.compressor = self.meta.get("compressor")

    def __getitem__(self, item):
        if isinstance(item, tuple):
            if len(item) == 0:
                raise IndexError("empty index")
            first = item[0]
            rest = item[1:]
        else:
            first = item
            rest = ()

        if not isinstance(first, (int, np.integer)):
            return self._read_full()[item]

        frame = int(first)
        if frame < 0:
            frame += self.shape[0]
        if frame < 0 or frame >= self.shape[0]:
            raise IndexError(frame)

        frame_array = self._read_frame(frame)
        return frame_array[rest] if rest else frame_array

    def _read_full(self) -> np.ndarray:
        out = np.full(self.shape, self.fill_value, dtype=self.dtype)
        grid_shape = tuple((s + c - 1) // c for s, c in zip(self.shape, self.chunks))
        for chunk_index in np.ndindex(*grid_shape):
            chunk = self._read_chunk(chunk_index)
            slices = self._chunk_slices(chunk_index)
            crop = tuple(slice(0, sl.stop - sl.start) for sl in slices)
            out[slices] = chunk[crop]
        return out

    def _read_frame(self, frame: int) -> np.ndarray:
        out = np.full(self.shape[1:], self.fill_value, dtype=self.dtype)
        first_chunk = frame // self.chunks[0]
        frame_offset = frame % self.chunks[0]
        spatial_grid_shape = tuple(
            (size + chunk - 1) // chunk
            for size, chunk in zip(self.shape[1:], self.chunks[1:])
        )

        for spatial_index in np.ndindex(*spatial_grid_shape):
            chunk_index = (first_chunk, *spatial_index)
            chunk = self._read_chunk(chunk_index)
            full_slices = self._chunk_slices(chunk_index)
            frame_in_chunk = chunk[frame_offset]
            out_slices = tuple(slice(sl.start, sl.stop) for sl in full_slices[1:])
            crop_slices = tuple(slice(0, sl.stop - sl.start) for sl in full_slices[1:])
            out[out_slices] = frame_in_chunk[crop_slices]

        return out

    def _chunk_slices(self, chunk_index: tuple[int, ...]) -> tuple[slice, ...]:
        slices = []
        for axis, chunk_number in enumerate(chunk_index):
            start = int(chunk_number) * self.chunks[axis]
            stop = min(start + self.chunks[axis], self.shape[axis])
            slices.append(slice(start, stop))
        return tuple(slices)

    def _read_chunk(self, chunk_index: tuple[int, ...]) -> np.ndarray:
        chunk_path = self.array_dir / ".".join(str(i) for i in chunk_index)
        if not chunk_path.exists():
            return np.full(self.chunks, self.fill_value, dtype=self.dtype)

        raw = self._decompress(chunk_path.read_bytes())
        arr = np.frombuffer(raw, dtype=self.dtype)
        expected = int(np.prod(self.chunks))
        if arr.size != expected:
            raise ValueError(
                f"Chunk {chunk_path} has {arr.size} values, expected {expected}."
            )
        return arr.reshape(self.chunks, order=self.order)

    def _decompress(self, raw: bytes) -> bytes:
        compressor = self.compressor
        if compressor in (None, False):
            return raw

        cname = compressor.get("id") if isinstance(compressor, dict) else str(compressor)
        if cname in {"gzip", "gz"}:
            return gzip.decompress(raw)
        if cname == "zlib":
            return zlib.decompress(raw)

        try:
            import numcodecs  # type: ignore

            return numcodecs.get_codec(compressor).decode(raw)
        except Exception as exc:
            raise RuntimeError(
                f"Unsupported Zarr compressor for offline fallback: {compressor!r}. "
                "Install zarr/numcodecs or use an uncompressed/gzip/zlib store."
            ) from exc


def open_array(zarr_path: str | Path, array_key: str = "0"):
    """Open ``array_key`` from a Zarr group.

    Returns an object with ``shape`` and integer frame indexing. Uses the real
    zarr package when available, otherwise uses ``MinimalZarrArray``.
    """

    zarr_path = Path(zarr_path)
    try:
        import zarr  # type: ignore

        return zarr.open_group(str(zarr_path), mode="r")[array_key]
    except ModuleNotFoundError:
        pass

    array_dir = zarr_path / array_key
    if not array_dir.exists() and (zarr_path / ".zarray").exists():
        array_dir = zarr_path
    return MinimalZarrArray(array_dir)
