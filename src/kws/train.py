"""Training loop.

Augmentation is re-applied every epoch, which means re-extracting MFCCs from
freshly shifted and noise-mixed waveforms each time. That is slower than
caching one augmented copy, and it is the point: the model should see a
distribution of distortions, not a fixed sample of them.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from kws.config import Config
from kws.data import LABELS, Example
from kws.dataset import FeatureSet, featurise
from kws.models import DSCNN, MFCCLogisticRegression, count_parameters


@dataclass
class TrainResult:
    model: nn.Module
    history: list[dict[str, float]]
    best_val_accuracy: float
    train_seconds: float
    normalisation: tuple[float, float]


def _tensor(x: np.ndarray) -> torch.Tensor:
    """Add the channel axis the convolutions expect: (N, 1, frames, mfcc)."""
    return torch.from_numpy(x.astype(np.float32)).unsqueeze(1)


@torch.no_grad()
def predict(model: nn.Module, x: np.ndarray, batch_size: int = 512) -> np.ndarray:
    model.eval()
    chunks = []
    for start in range(0, len(x), batch_size):
        chunks.append(model(_tensor(x[start : start + batch_size])).numpy())
    return np.concatenate(chunks).argmax(axis=1)


def build_model(cfg: Config, architecture: str = "dscnn") -> nn.Module:
    if architecture == "dscnn":
        return DSCNN(
            n_classes=len(LABELS),
            channels=cfg.model.channels,
            n_blocks=cfg.model.n_blocks,
            dropout=cfg.model.dropout,
        )
    if architecture == "linear":
        return MFCCLogisticRegression(n_mfcc=cfg.frontend.n_mfcc, n_classes=len(LABELS))
    raise ValueError(f"unknown architecture {architecture!r}")


def train(
    train_examples: list[Example],
    train_features: FeatureSet,
    val_features: FeatureSet,
    cfg: Config,
    normalisation: tuple[float, float],
    architecture: str = "dscnn",
    verbose: bool = True,
) -> TrainResult:
    """Fit a model, keeping the weights with the best validation accuracy."""
    torch.manual_seed(cfg.train.seed)
    np.random.seed(cfg.train.seed)
    rng = np.random.default_rng(cfg.train.seed)
    mean, std = normalisation

    model = build_model(cfg, architecture)
    optimiser = torch.optim.AdamW(
        model.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=cfg.train.epochs)
    criterion = nn.CrossEntropyLoss()

    history: list[dict[str, float]] = []
    best_accuracy, best_state, stale = -1.0, None, 0
    start = time.perf_counter()
    frontend = cfg.frontend_config()

    for epoch in range(1, cfg.train.epochs + 1):
        if cfg.train.augment:
            epoch_set = featurise(
                train_examples, cfg.data.root, frontend, augment_with=rng
            )
            epoch_x = (epoch_set.x - mean) / std
            epoch_y = epoch_set.y
        else:
            epoch_x, epoch_y = train_features.x, train_features.y

        loader = DataLoader(
            TensorDataset(_tensor(epoch_x), torch.from_numpy(epoch_y)),
            batch_size=cfg.train.batch_size,
            shuffle=True,
        )

        model.train()
        total_loss, seen = 0.0, 0
        for xb, yb in loader:
            optimiser.zero_grad(set_to_none=True)
            loss = criterion(model(xb), yb)
            loss.backward()
            optimiser.step()
            total_loss += loss.item() * len(yb)
            seen += len(yb)
        scheduler.step()

        accuracy = float(np.mean(predict(model, val_features.x) == val_features.y))
        history.append(
            {"epoch": epoch, "train_loss": total_loss / seen, "val_accuracy": accuracy}
        )
        if verbose:
            print(
                f"epoch {epoch:3d}  loss {total_loss / seen:.4f}  val_acc {accuracy:.4f}",
                flush=True,
            )

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= cfg.train.patience:
                if verbose:
                    print(f"early stop at epoch {epoch}", flush=True)
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    return TrainResult(
        model=model,
        history=history,
        best_val_accuracy=best_accuracy,
        train_seconds=time.perf_counter() - start,
        normalisation=normalisation,
    )


def save_checkpoint(
    result: TrainResult, cfg: Config, architecture: str, path: str | Path
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": result.model.state_dict(),
            "config": cfg.to_dict(),
            "architecture": architecture,
            "labels": list(LABELS),
            "normalisation": result.normalisation,
            "n_parameters": count_parameters(result.model),
            "best_val_accuracy": result.best_val_accuracy,
            "history": result.history,
        },
        path,
    )
    return path


def load_checkpoint(path: str | Path) -> tuple[nn.Module, dict[str, Any]]:
    blob = torch.load(path, map_location="cpu", weights_only=False)
    cfg = Config()
    for section, values in blob["config"].items():
        target = getattr(cfg, section)
        for key, value in values.items():
            setattr(target, key, value)
    model = build_model(cfg, blob["architecture"])
    model.load_state_dict(blob["state_dict"])
    model.eval()
    return model, blob
