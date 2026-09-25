"""PyTorch dataset for the first detector baseline."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset
import zarr

from configs.baseline import (
    CROP_SIZE_ZYX,
    IMAGE_ARRAY_KEY,
    POSITIVE_CROP_PROB,
    TARGET_SIGMA_UM,
    TEMPORAL_OFFSETS,
    VOXEL_SIZE_ZYX_UM,
)
from src.geff import load_gt
from src.preprocessing import choose_crop_start, crop_zyx, percentile_normalize
from src.targets import make_heatmap


class BiohubDetectorDataset(Dataset):
    """Sample temporal 3D crops and Gaussian center heatmaps."""

    def __init__(
        self,
        data_root: str | Path,
        sample_names: list[str],
        samples_per_epoch: int = 4096,
        crop_size_zyx: tuple[int, int, int] = CROP_SIZE_ZYX,
        positive_crop_prob: float = POSITIVE_CROP_PROB,
        seed: int = 42,
    ) -> None:
        self.data_root = Path(data_root)
        self.train_dir = self.data_root / "train"
        self.sample_names = list(sample_names)
        self.samples_per_epoch = int(samples_per_epoch)
        self.crop_size_zyx = crop_size_zyx
        self.positive_crop_prob = float(positive_crop_prob)
        self.seed = int(seed)

        self._metadata = {}
        for name in self.sample_names:
            nodes, _ = load_gt(self.train_dir / f"{name}.geff")
            self._metadata[name] = {
                "nodes": nodes,
                "frames": np.sort(nodes["t"].unique().astype(np.int64)),
            }

    def __len__(self) -> int:
        return self.samples_per_epoch

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        rng = np.random.default_rng(self.seed + int(index))
        name = self.sample_names[int(rng.integers(0, len(self.sample_names)))]
        meta = self._metadata[name]
        nodes = meta["nodes"]
        frames = meta["frames"]

        image = zarr.open_group(str(self.train_dir / f"{name}.zarr"), mode="r")[
            IMAGE_ARRAY_KEY
        ]

        t = int(frames[int(rng.integers(0, len(frames)))])
        frame_nodes = nodes[nodes["t"] == t]

        center = None
        if len(frame_nodes) and rng.random() < self.positive_crop_prob:
            row = frame_nodes.iloc[int(rng.integers(0, len(frame_nodes)))]
            center = (int(row["z"]), int(row["y"]), int(row["x"]))

        volume_shape = tuple(int(v) for v in image.shape[1:])
        crop_start = choose_crop_start(center, volume_shape, self.crop_size_zyx, rng)

        channels = []
        for offset in TEMPORAL_OFFSETS:
            tt = min(max(t + offset, 0), image.shape[0] - 1)
            volume = np.asarray(image[tt])
            crop = crop_zyx(volume, crop_start, self.crop_size_zyx)
            channels.append(percentile_normalize(crop))

        crop_nodes = self._nodes_in_crop(frame_nodes, crop_start)
        heatmap = make_heatmap(
            crop_nodes,
            self.crop_size_zyx,
            TARGET_SIGMA_UM,
            VOXEL_SIZE_ZYX_UM,
        )

        return {
            "image": torch.from_numpy(np.stack(channels, axis=0)).float(),
            "target": torch.from_numpy(heatmap[None]).float(),
        }

    def _nodes_in_crop(
        self,
        frame_nodes,
        crop_start_zyx: tuple[int, int, int],
    ) -> np.ndarray:
        z0, y0, x0 = crop_start_zyx
        dz, dy, dx = self.crop_size_zyx

        inside = frame_nodes[
            (frame_nodes["z"] >= z0)
            & (frame_nodes["z"] < z0 + dz)
            & (frame_nodes["y"] >= y0)
            & (frame_nodes["y"] < y0 + dy)
            & (frame_nodes["x"] >= x0)
            & (frame_nodes["x"] < x0 + dx)
        ]

        points = inside[["z", "y", "x"]].to_numpy(dtype=np.int64)
        points[:, 0] -= z0
        points[:, 1] -= y0
        points[:, 2] -= x0
        return points
