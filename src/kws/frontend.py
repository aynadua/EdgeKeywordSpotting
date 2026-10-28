"""The audio front end, written out rather than imported.

An always-on keyword spotter runs its feature extraction on the
microcontroller, continuously, on a battery. Whatever happens in the front end
has to be implementable in a few hundred lines of C against a CMSIS-DSP FFT.
Calling ``librosa.feature.mfcc`` hides exactly the decisions that determine
whether that is possible -- frame size, FFT length, filterbank layout, how
many coefficients survive -- so the pipeline is built here from the FFT up.

The configuration is the one ARM's "Hello Edge" work uses for MCU keyword
spotting: 40 ms frames, 20 ms hop, 40 mel bands, 10 DCT coefficients. One
second of 16 kHz audio becomes a 10x49 matrix -- 490 values, under 2 KB in
int8, which is a sane working set for a device with 64 KB of RAM.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np
from scipy.fftpack import dct


@dataclass(frozen=True)
class FrontendConfig:
    """Framing and filterbank geometry."""

    sample_rate: int = 16000
    window_ms: float = 40.0
    hop_ms: float = 20.0
    n_mels: int = 40
    n_mfcc: int = 10
    fmin: float = 20.0
    fmax: float = 7600.0
    pre_emphasis: float = 0.97
    log_floor: float = 1e-6

    @property
    def frame_length(self) -> int:
        return int(round(self.sample_rate * self.window_ms / 1000.0))

    @property
    def hop_length(self) -> int:
        return int(round(self.sample_rate * self.hop_ms / 1000.0))

    @property
    def n_fft(self) -> int:
        """Next power of two at or above the frame length."""
        return 1 << (self.frame_length - 1).bit_length()

    def n_frames(self, n_samples: int) -> int:
        if n_samples < self.frame_length:
            return 0
        return 1 + (n_samples - self.frame_length) // self.hop_length


def hz_to_mel(hz: np.ndarray | float) -> np.ndarray | float:
    """O'Shaughnessy's mel scale, the one HTK and most KWS work uses."""
    return 2595.0 * np.log10(1.0 + np.asarray(hz, dtype=np.float64) / 700.0)


def mel_to_hz(mel: np.ndarray | float) -> np.ndarray | float:
    return 700.0 * (10.0 ** (np.asarray(mel, dtype=np.float64) / 2595.0) - 1.0)


@lru_cache(maxsize=8)
def mel_filterbank(
    n_fft: int, sample_rate: int, n_mels: int, fmin: float, fmax: float
) -> np.ndarray:
    """Triangular mel filterbank, ``(n_mels, n_fft // 2 + 1)``.

    Filters are area-normalised so that a flat spectrum produces a flat
    filterbank output rather than one that rises with centre frequency.
    """
    n_bins = n_fft // 2 + 1
    edges = mel_to_hz(np.linspace(hz_to_mel(fmin), hz_to_mel(fmax), n_mels + 2))
    bins = np.floor((n_fft + 1) * edges / sample_rate).astype(int)
    bins = np.clip(bins, 0, n_bins - 1)

    filters = np.zeros((n_mels, n_bins), dtype=np.float64)
    for m in range(n_mels):
        left, centre, right = bins[m], bins[m + 1], bins[m + 2]
        if centre == left:
            centre = min(left + 1, n_bins - 1)
        if right == centre:
            right = min(centre + 1, n_bins - 1)
        if centre > left:
            filters[m, left:centre] = (np.arange(left, centre) - left) / (centre - left)
        if right > centre:
            filters[m, centre:right] = (right - np.arange(centre, right)) / (right - centre)

        width = edges[m + 2] - edges[m]
        if width > 0:
            filters[m] *= 2.0 / width
    return filters


def frame(signal: np.ndarray, frame_length: int, hop_length: int) -> np.ndarray:
    """Split into overlapping frames as a strided view -- no copy."""
    n = 1 + (len(signal) - frame_length) // hop_length if len(signal) >= frame_length else 0
    if n <= 0:
        return np.zeros((0, frame_length), dtype=signal.dtype)
    strides = (signal.strides[0] * hop_length, signal.strides[0])
    return np.lib.stride_tricks.as_strided(signal, shape=(n, frame_length), strides=strides)


def log_mel_spectrogram(waveform: np.ndarray, cfg: FrontendConfig) -> np.ndarray:
    """Log-mel spectrogram, ``(n_mels, n_frames)``."""
    x = np.asarray(waveform, dtype=np.float64)
    if cfg.pre_emphasis > 0 and len(x) > 1:
        x = np.append(x[0], x[1:] - cfg.pre_emphasis * x[:-1])

    frames = frame(np.ascontiguousarray(x), cfg.frame_length, cfg.hop_length)
    if not len(frames):
        return np.zeros((cfg.n_mels, 0))

    windowed = frames * np.hanning(cfg.frame_length)[None, :]
    power = np.abs(np.fft.rfft(windowed, n=cfg.n_fft, axis=1)) ** 2

    bank = mel_filterbank(cfg.n_fft, cfg.sample_rate, cfg.n_mels, cfg.fmin, cfg.fmax)
    return np.log(power @ bank.T + cfg.log_floor).T


def mfcc(waveform: np.ndarray, cfg: FrontendConfig | None = None) -> np.ndarray:
    """MFCC features, ``(n_mfcc, n_frames)``, float32.

    The DCT is orthonormal so coefficient magnitudes do not depend on the
    number of mel bands, which keeps the features stable if the filterbank is
    ever resized.
    """
    cfg = cfg or FrontendConfig()
    spectrogram = log_mel_spectrogram(waveform, cfg)
    if not spectrogram.shape[1]:
        return np.zeros((cfg.n_mfcc, 0), dtype=np.float32)
    coefficients = dct(spectrogram, type=2, axis=0, norm="ortho")[: cfg.n_mfcc]
    return coefficients.astype(np.float32)


def feature_shape(cfg: FrontendConfig, n_samples: int) -> tuple[int, int]:
    """Shape :func:`mfcc` will return for an input of this length."""
    return cfg.n_mfcc, cfg.n_frames(n_samples)
