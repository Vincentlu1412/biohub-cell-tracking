"""Target generation for detector training."""

from __future__ import annotations

import numpy as np


def sigma_um_to_voxels(
    sigma_um: float,
    voxel_size_zyx_um: tuple[float, float, float],
) -> tuple[float, float, float]:
    """Convert an isotropic physical sigma to anisotropic voxel sigmas."""

    return tuple(float(sigma_um / spacing) for spacing in voxel_size_zyx_um)


def draw_gaussian_3d(
    heatmap: np.ndarray,
    center_zyx: tuple[int, int, int],
    sigma_um: float,
    voxel_size_zyx_um: tuple[float, float, float],
) -> None:
    """Draw one anisotropic 3D Gaussian into heatmap using max composition."""

    z0, y0, x0 = [int(v) for v in center_zyx]
    sz, sy, sx = sigma_um_to_voxels(sigma_um, voxel_size_zyx_um)

    rz = max(1, int(np.ceil(3 * sz)))
    ry = max(1, int(np.ceil(3 * sy)))
    rx = max(1, int(np.ceil(3 * sx)))

    depth, height, width = heatmap.shape
    z1, z2 = max(0, z0 - rz), min(depth, z0 + rz + 1)
    y1, y2 = max(0, y0 - ry), min(height, y0 + ry + 1)
    x1, x2 = max(0, x0 - rx), min(width, x0 + rx + 1)

    if z1 >= z2 or y1 >= y2 or x1 >= x2:
        return

    zz, yy, xx = np.meshgrid(
        np.arange(z1, z2),
        np.arange(y1, y2),
        np.arange(x1, x2),
        indexing="ij",
    )

    gaussian = np.exp(
        -0.5
        * (
            ((zz - z0) / sz) ** 2
            + ((yy - y0) / sy) ** 2
            + ((xx - x0) / sx) ** 2
        )
    )

    patch = heatmap[z1:z2, y1:y2, x1:x2]
    np.maximum(patch, gaussian.astype(np.float32), out=patch)


def make_heatmap(
    points_zyx: np.ndarray,
    shape_zyx: tuple[int, int, int],
    sigma_um: float,
    voxel_size_zyx_um: tuple[float, float, float],
) -> np.ndarray:
    """Create a detector heatmap for all points inside one crop."""

    heatmap = np.zeros(shape_zyx, dtype=np.float32)

    for point in points_zyx:
        draw_gaussian_3d(
            heatmap=heatmap,
            center_zyx=(int(point[0]), int(point[1]), int(point[2])),
            sigma_um=sigma_um,
            voxel_size_zyx_um=voxel_size_zyx_um,
        )

    return heatmap
