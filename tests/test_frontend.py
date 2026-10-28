"""The audio front end, checked against signals whose answer is known.

Because the front end is written from the FFT up rather than imported, it has
to be verified against physics rather than against another library. A pure
tone must light up the mel band containing its frequency; a DC signal must
produce no AC energy; the geometry must produce exactly the frame count the
MCU budget assumes.
"""

import numpy as np
import pytest

from kws.frontend import (
    FrontendConfig,
    feature_shape,
    frame,
    hz_to_mel,
    log_mel_spectrogram,
    mel_filterbank,
    mel_to_hz,
    mfcc,
)

CFG = FrontendConfig()
SR = 16000


def _tone(hz: float, seconds: float = 1.0) -> np.ndarray:
    t = np.arange(int(SR * seconds)) / SR
    return np.sin(2 * np.pi * hz * t).astype(np.float32)


def test_geometry_matches_the_mcu_budget():
    """40 ms frames, 20 ms hop, one second of audio -> a 10x49 matrix."""
    assert CFG.frame_length == 640
    assert CFG.hop_length == 320
    assert CFG.n_fft == 1024
    assert CFG.n_frames(SR) == 49
    assert feature_shape(CFG, SR) == (10, 49)


def test_mel_scale_round_trips():
    for hz in (20.0, 440.0, 1000.0, 7600.0):
        assert mel_to_hz(hz_to_mel(hz)) == pytest.approx(hz, rel=1e-9)


def test_mel_scale_is_monotonic_and_compresses_high_frequencies():
    assert hz_to_mel(200) - hz_to_mel(100) > hz_to_mel(5100) - hz_to_mel(5000)


def test_filterbank_shape_and_coverage():
    bank = mel_filterbank(1024, SR, 40, 20.0, 7600.0)
    assert bank.shape == (40, 513)
    assert np.all(bank >= 0)
    assert np.all(bank.sum(axis=1) > 0), "every filter must cover at least one bin"


@pytest.mark.parametrize("hz", [250.0, 1000.0, 3000.0])
def test_a_pure_tone_peaks_in_the_band_containing_it(hz):
    spectrogram = log_mel_spectrogram(_tone(hz), CFG)
    peak_band = int(np.argmax(spectrogram.mean(axis=1)))

    centres = mel_to_hz(np.linspace(hz_to_mel(CFG.fmin), hz_to_mel(CFG.fmax), CFG.n_mels + 2))[1:-1]
    assert abs(centres[peak_band] - hz) < 0.25 * hz


def test_framing_is_a_view_with_the_right_count():
    signal = np.arange(1000, dtype=np.float64)
    frames = frame(signal, 100, 50)
    assert frames.shape == (19, 100)
    assert np.array_equal(frames[0], signal[:100])
    assert np.array_equal(frames[1], signal[50:150])


def test_framing_handles_a_signal_shorter_than_one_frame():
    assert frame(np.zeros(10), 100, 50).shape == (0, 100)


def test_mfcc_shape_and_dtype():
    coefficients = mfcc(_tone(1000.0), CFG)
    assert coefficients.shape == (10, 49)
    assert coefficients.dtype == np.float32
    assert np.all(np.isfinite(coefficients))


def test_mfcc_of_silence_is_finite():
    """The log floor must keep digital silence from producing -inf."""
    assert np.all(np.isfinite(mfcc(np.zeros(SR, dtype=np.float32), CFG)))


def test_gain_shifts_only_c0_when_every_band_carries_energy():
    """C0 carries overall level; the higher coefficients describe shape.

    Scaling a waveform multiplies every mel band's power by the same factor,
    which is a constant offset after the log -- and a constant offset lands
    entirely in the DCT's DC term, C0. That holds exactly when every band has
    real energy in it, so the test uses broadband noise.
    """
    noise = np.random.default_rng(0).standard_normal(SR).astype(np.float32) * 0.3
    loud = mfcc(noise, CFG)
    quiet = mfcc(noise * 0.5, CFG)

    assert quiet[0].mean() < loud[0].mean() - 1.0
    assert np.allclose(quiet[1:], loud[1:], atol=0.02)


def test_gain_invariance_breaks_down_on_real_speech():
    """A documented limitation, pinned so it cannot regress silently.

    Speech does not excite every mel band. In the bands that carry almost
    nothing, the log is dominated by ``log_floor`` rather than by the signal,
    and the floor does not scale with the waveform. So halving the gain of a
    real clip changes the *shape* of the log-mel vector, not just its level,
    and the change leaks into C1 and above.

    The practical consequence is that this front end is not level-invariant:
    a model trained on clips recorded at one gain degrades at another. Mean
    normalisation per utterance, or AGC ahead of the front end, is the usual
    fix and is not implemented here.
    """
    rng = np.random.default_rng(1)
    speech_like = rng.standard_normal(SR).astype(np.float32) * 0.3
    # Silence most of it, as a spoken word is mostly silence.
    speech_like[: int(0.35 * SR)] = 0.0
    speech_like[int(0.65 * SR) :] = 0.0

    loud = mfcc(speech_like, CFG)
    quiet = mfcc(speech_like * 0.25, CFG)
    assert np.abs(quiet[1:] - loud[1:]).max() > 0.1


def test_an_empty_signal_gives_no_frames():
    assert mfcc(np.zeros(10, dtype=np.float32), CFG).shape == (10, 0)


def test_a_different_hop_changes_the_frame_count():
    faster = FrontendConfig(hop_ms=10.0)
    assert faster.n_frames(SR) > CFG.n_frames(SR)
