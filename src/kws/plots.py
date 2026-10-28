"""Figures for the README. Headless by construction."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


def confusion(record: dict[str, Any], path: str | Path) -> Path:
    """Row-normalised confusion matrix over the 12 classes."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    matrix = np.array(record["confusion_matrix"])
    labels = record["labels"]

    fig, ax = plt.subplots(figsize=(7, 6.2), dpi=140)
    image = ax.imshow(matrix, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(labels)), labels, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(labels)), labels, fontsize=8)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"{record['model']}: test confusion", fontsize=11)

    for i, row in enumerate(matrix):
        for j, value in enumerate(row):
            if value >= 0.01:
                ax.text(
                    j, i, f"{value:.2f}", ha="center", va="center",
                    color="white" if value > 0.5 else "black", fontsize=6.5,
                )

    fig.colorbar(image, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


def budget(records: list[dict[str, Any]], path: str | Path) -> Path:
    """Accuracy against model size, with the MCU budget marked."""
    from kws.export import LATENCY_BUDGET_MS, SIZE_BUDGET_KB

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    scored = [r for r in records if "size_kb" in r]
    if not scored:
        return path

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), dpi=140)
    for record in scored:
        axes[0].scatter(record["size_kb"], record["accuracy"], s=70)
        axes[0].annotate(
            record["model"], (record["size_kb"], record["accuracy"]),
            textcoords="offset points", xytext=(6, 5), fontsize=8,
        )
        axes[1].scatter(record["latency_ms_p50"], record["accuracy"], s=70)
        axes[1].annotate(
            record["model"], (record["latency_ms_p50"], record["accuracy"]),
            textcoords="offset points", xytext=(6, 5), fontsize=8,
        )

    axes[0].axvline(SIZE_BUDGET_KB, linestyle="--", color="tab:red", linewidth=1)
    axes[0].text(SIZE_BUDGET_KB, axes[0].get_ylim()[0], " budget", color="tab:red", fontsize=8)
    axes[0].set_xlabel("artefact size (KB)")
    axes[1].axvline(LATENCY_BUDGET_MS, linestyle="--", color="tab:red", linewidth=1)
    axes[1].set_xlabel("p50 latency (ms, one core)")

    for ax in axes:
        ax.set_ylabel("test accuracy")
        ax.grid(alpha=0.3)
    fig.suptitle("Accuracy against the microcontroller budget", fontsize=11)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


def features(path: str | Path, root: str, n_examples: int = 4) -> Path:
    """What the model sees: waveform and MFCC for a few keywords."""
    from kws.data import read_clip
    from kws.frontend import FrontendConfig, log_mel_spectrogram, mfcc

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    cfg = FrontendConfig()
    words = ["yes", "no", "stop", "go"][:n_examples]

    fig, axes = plt.subplots(3, len(words), figsize=(3.1 * len(words), 7), dpi=130)
    for column, word in enumerate(words):
        candidates = sorted((Path(root) / word).glob("*.wav"))
        if not candidates:
            continue
        waveform = read_clip(candidates[0])

        axes[0, column].plot(np.arange(len(waveform)) / 16000, waveform, linewidth=0.4)
        axes[0, column].set_title(f'"{word}"', fontsize=11)
        axes[0, column].set_xlim(0, 1)
        if column == 0:
            axes[0, column].set_ylabel("waveform")

        axes[1, column].imshow(
            log_mel_spectrogram(waveform, cfg), aspect="auto", origin="lower", cmap="magma"
        )
        if column == 0:
            axes[1, column].set_ylabel("log-mel (40 bands)")

        axes[2, column].imshow(mfcc(waveform, cfg), aspect="auto", origin="lower", cmap="viridis")
        axes[2, column].set_xlabel("frame")
        if column == 0:
            axes[2, column].set_ylabel("MFCC (10 coeffs)")

    fig.suptitle("Front end: 1 s of audio becomes a 10x49 MFCC matrix", fontsize=11)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


def training_curve(history: list[dict[str, float]], path: str | Path) -> Path:
    """Loss and validation accuracy per epoch."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    epochs = [h["epoch"] for h in history]
    fig, ax = plt.subplots(figsize=(7, 4), dpi=140)
    ax.plot(epochs, [h["train_loss"] for h in history], label="train loss", color="tab:blue")
    ax.set_xlabel("epoch")
    ax.set_ylabel("training loss", color="tab:blue")

    twin = ax.twinx()
    twin.plot(
        epochs, [h["val_accuracy"] for h in history], label="val accuracy", color="tab:orange"
    )
    twin.set_ylabel("validation accuracy", color="tab:orange")

    ax.set_title("Training")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path
