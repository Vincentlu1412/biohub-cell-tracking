"""Image preprocessing utilities."""

from __future__ import annotations

import numpy as np


def percentile_normalize(
    image: np.ndarray,
    lower: float = 1.0,
    upper: float = 99.8,
    eps: float = 1e-6,
) -> np.ndarray:
    """Normalize an image volume to roughly [0, 1] using robust percentiles."""

    image = image.astype(np.float32, copy=False)
    lo, hi = np.percentile(image, [lower, upper])

    if hi <= lo + eps:
        return np.zeros_like(image, dtype=np.float32)

    image = (image - lo) / (hi - lo + eps)
    return np.clip(image, 0.0, 1.0).astype(np.float32, copy=False)


def pad_to_shape_zyx(volume: np.ndarray, shape_zyx: tuple[int, int, int]) -> np.ndarray:
    """Pad a ZYX volume at the end of each axis until it reaches shape_zyx."""

    pad_width = []
    for current, target in zip(volume.shape, shape_zyx):
        pad_width.append((0, max(0, target - current)))

    if all(before == 0 and after == 0 for before, after in pad_width):
        return volume

    return np.pad(volume, pad_width, mode="edge")


def crop_zyx(
    volume: np.ndarray,
    start_zyx: tuple[int, int, int],
    crop_size_zyx: tuple[int, int, int],
) -> np.ndarray:
    """Crop a ZYX volume, padding first if the source is smaller than the crop."""

    volume = pad_to_shape_zyx(volume, crop_size_zyx)
    z0, y0, x0 = start_zyx
    dz, dy, dx = crop_size_zyx

    return volume[z0 : z0 + dz, y0 : y0 + dy, x0 : x0 + dx]


def choose_crop_start(
    center_zyx: tuple[int, int, int] | None,
    volume_shape_zyx: tuple[int, int, int],
    crop_size_zyx: tuple[int, int, int],
    rng: np.random.Generator,
) -> tuple[int, int, int]:
    """Choose a valid crop start, optionally centered near a GT point."""

    starts = []
    for axis, (dim, crop) in enumerate(zip(volume_shape_zyx, crop_size_zyx)):
        max_start = max(0, dim - crop)

        if center_zyx is None:
            start = int(rng.integers(0, max_start + 1)) if max_start > 0 else 0
        else:
            center = int(center_zyx[axis])
            low = max(0, center - crop + 1)
            high = min(center, max_start)
            start = int(rng.integers(low, high + 1)) if high >= low else 0

        starts.append(start)

    return tuple(starts)
