"""Turning the corpus into cached MFCC tensors.

Feature extraction runs once and is cached to ``.npz``. It is deterministic
given the split spec, so a cached file and a fresh extraction agree.

Training-time augmentation -- time shift and background noise mixing -- is
applied on the fly to the *waveform*, not the cached features, because the
point of the augmentation is to make the front end see a different signal.
Caching augmented features would mean training on a handful of fixed
distortions instead of a distribution of them.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from kws.data import LABEL_TO_INDEX, SAMPLE_RATE, Example, load_noise, render
from kws.frontend import FrontendConfig, mfcc


@dataclass
class FeatureSet:
    """Cached features and labels for one split."""

    x: np.ndarray  # (N, n_frames, n_mfcc) float32
    y: np.ndarray  # (N,) int64

    def __len__(self) -> int:
        return len(self.y)

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, x=self.x, y=self.y)
        return path

    @classmethod
    def load(cls, path: str | Path) -> FeatureSet:
        with np.load(path) as f:
            return cls(x=f["x"], y=f["y"])


def augment(
    waveform: np.ndarray,
    noise: list[np.ndarray],
    rng: np.random.Generator,
    shift_ms: float = 100.0,
    noise_probability: float = 0.8,
    noise_gain: float = 0.1,
) -> np.ndarray:
    """Random time shift plus background noise, as the reference recipe does.

    Both matter for a wake word: the keyword never starts exactly at the
    beginning of the window, and a device listening in a kitchen hears a
    kitchen.
    """
    length = len(waveform)
    max_shift = int(SAMPLE_RATE * shift_ms / 1000)
    shift = int(rng.integers(-max_shift, max_shift + 1))
    shifted = np.zeros_like(waveform)
    if shift >= 0:
        shifted[shift:] = waveform[: length - shift]
    else:
        shifted[:shift] = waveform[-shift:]

    if noise and rng.random() < noise_probability:
        clip = noise[int(rng.integers(0, len(noise)))]
        if len(clip) > length:
            start = int(rng.integers(0, len(clip) - length))
            background = clip[start : start + length]
            shifted = shifted + background * float(rng.uniform(0.0, noise_gain))

    return np.clip(shifted, -1.0, 1.0).astype(np.float32)


def featurise(
    examples: list[Example],
    root: str | Path,
    cfg: FrontendConfig | None = None,
    augment_with: np.random.Generator | None = None,
) -> FeatureSet:
    """Extract MFCCs for a list of examples.

    Deliberately serial. Augmentation re-runs every epoch, so this is on the
    training hot path and parallelising it looks obviously worthwhile -- but
    measured on 6,000 clips across 4 cores it is 15.8 s with a process pool
    against 5.4 s without. Spawning workers, pickling the example list and
    re-reading the background recordings in every worker cost far more than
    the 34 s a full serial pass over 37,607 clips takes. The simple loop wins,
    so the simple loop is what is here.
    """
    cfg = cfg or FrontendConfig()
    noise = load_noise(root)
    n_frames = cfg.n_frames(SAMPLE_RATE)

    x = np.empty((len(examples), n_frames, cfg.n_mfcc), dtype=np.float32)
    y = np.empty(len(examples), dtype=np.int64)

    for i, example in enumerate(examples):
        waveform = render(example, noise, SAMPLE_RATE)
        if augment_with is not None:
            waveform = augment(waveform, noise, augment_with)
        x[i] = mfcc(waveform, cfg).T
        y[i] = LABEL_TO_INDEX[example.label]

    return FeatureSet(x=x, y=y)


def normalise(
    train: FeatureSet, *others: FeatureSet
) -> tuple[tuple[float, float], list[FeatureSet]]:
    """Standardise with statistics from the training split only.

    Computing the mean and standard deviation over the whole dataset is a
    small but real leak: the test split would influence the scaling the model
    is fitted under. The constants are also what has to be burned into the
    firmware, so they are returned rather than hidden.
    """
    mean = float(train.x.mean())
    std = float(train.x.std()) or 1.0

    scaled = [FeatureSet(x=(train.x - mean) / std, y=train.y)]
    scaled.extend(FeatureSet(x=(other.x - mean) / std, y=other.y) for other in others)
    return (mean, std), scaled
