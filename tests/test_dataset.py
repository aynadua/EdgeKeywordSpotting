"""Feature caching, augmentation and normalisation."""

from pathlib import Path

import numpy as np
import pytest

from kws.data import LABEL_TO_INDEX, SAMPLE_RATE, Example, build_splits, load_noise
from kws.dataset import FeatureSet, augment, featurise, normalise
from kws.frontend import FrontendConfig

ROOT = Path("data/speech_commands")
pytestmark = pytest.mark.skipif(
    not (ROOT / "yes").is_dir(), reason="Speech Commands not present; run `kws data`"
)


@pytest.fixture(scope="module")
def examples():
    return build_splits(ROOT).val[:120]


def test_featurise_shape_and_labels(examples):
    cfg = FrontendConfig()
    features = featurise(examples, ROOT, cfg)
    assert features.x.shape == (len(examples), cfg.n_frames(SAMPLE_RATE), cfg.n_mfcc)
    assert features.y.tolist() == [LABEL_TO_INDEX[e.label] for e in examples]
    assert np.all(np.isfinite(features.x))


def test_featurise_without_augmentation_is_deterministic(examples):
    a = featurise(examples, ROOT)
    b = featurise(examples, ROOT)
    assert np.array_equal(a.x, b.x)


def test_augmentation_changes_the_features_but_not_the_labels(examples):
    plain = featurise(examples, ROOT)
    noisy = featurise(examples, ROOT, augment_with=np.random.default_rng(0))
    assert not np.allclose(plain.x, noisy.x)
    assert np.array_equal(plain.y, noisy.y)


def test_augmentation_is_reproducible_for_a_given_seed(examples):
    a = featurise(examples, ROOT, augment_with=np.random.default_rng(5))
    b = featurise(examples, ROOT, augment_with=np.random.default_rng(5))
    assert np.array_equal(a.x, b.x)


def test_augmentation_differs_between_epochs(examples):
    """A single generator must give a different distortion each call."""
    rng = np.random.default_rng(1)
    first = featurise(examples, ROOT, augment_with=rng)
    second = featurise(examples, ROOT, augment_with=rng)
    assert not np.allclose(first.x, second.x)


def test_augment_preserves_length_and_range():
    rng = np.random.default_rng(2)
    waveform = rng.standard_normal(SAMPLE_RATE).astype(np.float32) * 0.2
    augmented = augment(waveform, load_noise(ROOT), rng)
    assert augmented.shape == waveform.shape
    assert np.abs(augmented).max() <= 1.0


def test_augment_with_no_noise_still_shifts():
    rng = np.random.default_rng(3)
    waveform = np.linspace(0, 1, SAMPLE_RATE).astype(np.float32)
    assert not np.allclose(augment(waveform, [], rng), waveform)


def test_normalisation_uses_training_statistics_only():
    """Fitting the scaler on test data would leak it into training."""
    train = FeatureSet(x=np.full((4, 2, 3), 10.0, dtype=np.float32), y=np.zeros(4, dtype=np.int64))
    test = FeatureSet(x=np.full((2, 2, 3), 50.0, dtype=np.float32), y=np.zeros(2, dtype=np.int64))

    (mean, std), (train_scaled, test_scaled) = normalise(train, test)
    assert mean == pytest.approx(10.0)
    # Test data is scaled by the training constants, not its own.
    assert test_scaled.x.mean() > train_scaled.x.mean()


def test_normalisation_centres_the_training_split():
    rng = np.random.default_rng(4)
    train = FeatureSet(
        x=rng.normal(5, 2, (200, 4, 3)).astype(np.float32), y=np.zeros(200, dtype=np.int64)
    )
    _, (scaled,) = normalise(train)
    assert scaled.x.mean() == pytest.approx(0.0, abs=1e-5)
    assert scaled.x.std() == pytest.approx(1.0, abs=1e-5)


def test_round_trip_through_disk(tmp_path, examples):
    features = featurise(examples[:10], ROOT)
    features.save(tmp_path / "f.npz")
    loaded = FeatureSet.load(tmp_path / "f.npz")
    assert np.array_equal(loaded.x, features.x)
    assert np.array_equal(loaded.y, features.y)


def test_empty_example_list_is_handled():
    assert len(featurise([], ROOT)) == 0


def test_silence_example_needs_no_file():
    example = Example(label="_silence_", noise_index=0, noise_offset=1000, gain=0.5)
    features = featurise([example], ROOT)
    assert features.x.shape[0] == 1
    assert np.all(np.isfinite(features.x))
