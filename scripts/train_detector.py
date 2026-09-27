"""Train the first 3D detector baseline."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from configs.baseline import DEFAULT_DATA_ROOT, RANDOM_SEED, VAL_FRACTION  # noqa: E402
from src.data_paths import resolve_data_root  # noqa: E402
from src.dataset import BiohubDetectorDataset  # noqa: E402
from src.losses import HeatmapLoss  # noqa: E402
from src.model import build_model  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Biohub detector baseline.")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--samples-per-epoch", type=int, default=2048)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "outputs/checkpoints")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_root = resolve_data_root(args.data_root)
    train_dir = data_root / "train"
    sample_names = sorted(path.stem for path in train_dir.glob("*.geff"))

    if not sample_names:
        raise FileNotFoundError(f"No GEFF files found under {train_dir}")

    train_names, val_names = train_test_split(
        sample_names,
        test_size=VAL_FRACTION,
        random_state=RANDOM_SEED,
    )

    train_dataset = BiohubDetectorDataset(
        data_root,
        train_names,
        samples_per_epoch=args.samples_per_epoch,
        seed=RANDOM_SEED,
    )
    val_dataset = BiohubDetectorDataset(
        data_root,
        val_names,
        samples_per_epoch=max(128, args.samples_per_epoch // 8),
        seed=RANDOM_SEED + 100_000,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model().to(device)
    criterion = HeatmapLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    best_val = float("inf")

    print(f"Device: {device}")
    print(f"Train samples: {len(train_names)}")
    print(f"Val samples  : {len(val_names)}")

    for epoch in range(1, args.epochs + 1):
        train_loss = run_epoch(model, train_loader, criterion, optimizer, device, train=True)
        val_loss = run_epoch(model, val_loader, criterion, optimizer, device, train=False)

        print(f"Epoch {epoch:03d} | train_loss={train_loss:.5f} | val_loss={val_loss:.5f}")

        last_path = args.output_dir / "detector_last.pt"
        torch.save(
            {
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "train_loss": train_loss,
                "val_loss": val_loss,
            },
            last_path,
        )

        if val_loss < best_val:
            best_val = val_loss
            torch.save(
                {
                    "epoch": epoch,
                    "model": model.state_dict(),
                    "val_loss": val_loss,
                },
                args.output_dir / "detector_best.pt",
            )


def run_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    criterion: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    train: bool,
) -> float:
    model.train(train)
    total_loss = 0.0

    context = torch.enable_grad() if train else torch.no_grad()
    with context:
        for batch in tqdm(loader, desc="train" if train else "val"):
            image = batch["image"].to(device, non_blocking=True)
            target = batch["target"].to(device, non_blocking=True)

            logits = model(image)
            loss = criterion(logits, target)

            if train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

            total_loss += float(loss.detach().cpu())

    return total_loss / max(1, len(loader))


if __name__ == "__main__":
    main()
