"""Dataset assembly.

The headline test is :func:`test_no_speaker_appears_in_two_splits`. Speaker
leakage is the defect that makes Speech Commands results look better than they
are, and it is invisible in any accuracy number.
"""

from pathlib import Path

import numpy as np
import pytest

from kws.data import (
    LABELS,
    SILENCE,
    TARGET_WORDS,
    UNKNOWN,
    build_splits,
    label_distribution,
    load_noise,
    read_clip,
    render,
    speaker_id,
    speakers_in,
)

ROOT = Path("data/speech_commands")
pytestmark = pytest.mark.skipif(
    not (ROOT / "yes").is_dir(), reason="Speech Commands not present; run `kws data`"
)


@pytest.fixture(scope="module")
def spec():
    return build_splits(ROOT)


def test_label_set_is_the_twelve_class_task():
    assert len(LABELS) == 12
    assert LABELS[0] == SILENCE and LABELS[1] == UNKNOWN
    assert set(LABELS[2:]) == set(TARGET_WORDS)


def test_no_speaker_appears_in_two_splits(spec):
    """Official lists partition by speaker hash; verify they really do."""
    train, val, test = (speakers_in(getattr(spec, s)) for s in ("train", "val", "test"))
    assert train and val and test
    assert not train & val
    assert not train & test
    assert not val & test


def test_splits_are_non_empty_and_ordered_by_size(spec):
    counts = spec.counts()
    assert counts["train"] > counts["test"] > 0
    assert counts["val"] > 0


@pytest.mark.parametrize("split", ["train", "val", "test"])
def test_every_label_is_present_in_every_split(spec, split):
    distribution = label_distribution(getattr(spec, split))
    assert all(count > 0 for count in distribution.values()), distribution


def test_target_classes_are_roughly_balanced(spec):
    distribution = label_distribution(spec.train)
    counts = [distribution[w] for w in TARGET_WORDS]
    assert max(counts) / min(counts) < 1.4


def test_unknown_and_silence_are_minority_classes(spec):
    distribution = label_distribution(spec.train)
    total = sum(distribution.values())
    assert 0.05 < distribution[UNKNOWN] / total < 0.15
    assert 0.05 < distribution[SILENCE] / total < 0.15


def test_speaker_id_is_extracted_from_the_filename():
    assert speaker_id(Path("yes/004ae714_nohash_0.wav")) == "004ae714"


def test_clips_are_padded_or_trimmed_to_exactly_one_second():
    path = sorted((ROOT / "yes").glob("*.wav"))[0]
    waveform = read_clip(path)
    assert waveform.shape == (16000,)
    assert waveform.dtype == np.float32
    assert np.abs(waveform).max() <= 1.0


def test_silence_examples_render_from_background_noise(spec):
    noise = load_noise(ROOT)
    silence = [e for e in spec.train if e.label == SILENCE][:5]
    assert silence
    for example in silence:
        waveform = render(example, noise)
        assert waveform.shape == (16000,)
        assert np.all(np.isfinite(waveform))


def test_silence_examples_vary(spec):
    """Identical silence clips would be a single example repeated thousands of times."""
    noise = load_noise(ROOT)
    silence = [e for e in spec.train if e.label == SILENCE][:4]
    rendered = [render(e, noise) for e in silence]
    assert not np.allclose(rendered[0], rendered[1])


def test_split_assembly_is_reproducible():
    a, b = build_splits(ROOT, seed=3), build_splits(ROOT, seed=3)
    assert [e.label for e in a.train[:200]] == [e.label for e in b.train[:200]]
