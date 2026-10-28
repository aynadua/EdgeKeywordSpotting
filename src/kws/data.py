"""Google Speech Commands V2, assembled as the standard 12-class task.

The benchmark everybody quotes is not "35-way word classification". It is
**V2-12**: ten target keywords, plus ``_silence_``, plus ``_unknown_`` for
every other word. A wake-word system has to reject the twenty-five words it
was not trained to spot and the silence between them, and a 35-way accuracy
quietly skips both halves of that job.

**Splits come from the dataset's own lists.** ``validation_list.txt`` and
``testing_list.txt`` ship with the corpus and partition it by a hash of the
*speaker* id, so no speaker appears in two splits. Splitting at random
instead -- which is easy, and common -- lets the same voice saying the same
word land in both train and test, and inflates accuracy by several points
while measuring speaker memorisation rather than keyword recognition.

``_silence_`` is built the way the reference implementation builds it: random
one-second crops of the six supplied background recordings, at randomised
gain. ``_unknown_`` is subsampled, because the twenty-five non-target words
outnumber the ten targets and an unbalanced ``_unknown_`` class teaches the
model to answer "unknown" to everything.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy.io import wavfile

DOWNLOAD_URL = "https://storage.googleapis.com/download.tensorflow.org/data/speech_commands_v0.02.tar.gz"

TARGET_WORDS: tuple[str, ...] = (
    "yes", "no", "up", "down", "left", "right", "on", "off", "stop", "go",
)
SILENCE = "_silence_"
UNKNOWN = "_unknown_"
LABELS: tuple[str, ...] = (SILENCE, UNKNOWN, *TARGET_WORDS)
LABEL_TO_INDEX = {label: i for i, label in enumerate(LABELS)}

NOISE_DIR = "_background_noise_"
SAMPLE_RATE = 16000


@dataclass
class Example:
    """One labelled clip: a path, or a crop of a noise recording."""

    label: str
    path: Path | None = None
    noise_index: int | None = None
    noise_offset: int = 0
    gain: float = 1.0


@dataclass
class SplitSpec:
    """The three splits, as lists of examples."""

    train: list[Example] = field(default_factory=list)
    val: list[Example] = field(default_factory=list)
    test: list[Example] = field(default_factory=list)

    def counts(self) -> dict[str, int]:
        return {"train": len(self.train), "val": len(self.val), "test": len(self.test)}


def download(root: str | Path) -> Path:
    """Fetch and unpack the corpus if it is not already present."""
    import tarfile
    from urllib.request import urlopen

    root = Path(root)
    if (root / "yes").is_dir():
        return root

    root.mkdir(parents=True, exist_ok=True)
    archive = root.parent / "speech_commands_v0.02.tar.gz"
    if not archive.exists():
        with urlopen(DOWNLOAD_URL, timeout=600) as response:  # noqa: S310
            archive.write_bytes(response.read())
    with tarfile.open(archive) as tar:
        tar.extractall(root, filter="data")
    return root


def speaker_id(path: Path) -> str:
    """The speaker hash embedded in a Speech Commands filename."""
    match = re.match(r"([^_]+)_nohash_", path.name)
    return match.group(1) if match else path.stem


def _read_list(path: Path) -> set[str]:
    if not path.exists():
        return set()
    return {line.strip() for line in path.read_text().splitlines() if line.strip()}


def load_noise(root: str | Path) -> list[np.ndarray]:
    """The background recordings, as float32 in [-1, 1].

    The supplied files carry extra RIFF chunks that SciPy warns about and then
    correctly ignores; the warning is suppressed rather than printed 47,000
    times.
    """
    import warnings

    root = Path(root) / NOISE_DIR
    clips = []
    for path in sorted(root.glob("*.wav")):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", wavfile.WavFileWarning)
            _, samples = wavfile.read(path)
        clips.append(samples.astype(np.float32) / 32768.0)
    return clips


def read_clip(path: str | Path, length: int = SAMPLE_RATE) -> np.ndarray:
    """Read a wav as float32 and pad or trim it to exactly ``length`` samples."""
    _, samples = wavfile.read(path)
    waveform = samples.astype(np.float32) / 32768.0
    if len(waveform) >= length:
        return waveform[:length]
    out = np.zeros(length, dtype=np.float32)
    out[: len(waveform)] = waveform
    return out


def render(example: Example, noise: list[np.ndarray], length: int = SAMPLE_RATE) -> np.ndarray:
    """Turn an example into a one-second waveform."""
    if example.path is not None:
        return read_clip(example.path, length)
    clip = noise[example.noise_index or 0]
    start = example.noise_offset % max(len(clip) - length, 1)
    return (clip[start : start + length] * example.gain).astype(np.float32)


def _which_split(relative: str, validation: set[str], testing: set[str]) -> str:
    if relative in validation:
        return "val"
    if relative in testing:
        return "test"
    return "train"


def _deterministic_unit(key: str) -> float:
    """A stable pseudo-random number in [0, 1) derived from a string."""
    digest = hashlib.sha1(key.encode()).hexdigest()  # noqa: S324 - not security
    return int(digest[:8], 16) / 0xFFFFFFFF


def build_splits(
    root: str | Path,
    unknown_fraction: float = 0.1,
    silence_fraction: float = 0.1,
    seed: int = 7,
) -> SplitSpec:
    """Assemble the 12-class task from the corpus on disk.

    Args:
        root: the unpacked dataset directory.
        unknown_fraction: how much of each split ``_unknown_`` should make up.
        silence_fraction: likewise for ``_silence_``.
        seed: controls unknown subsampling and silence crops only; split
            membership is always taken from the dataset's own lists.
    """
    root = Path(root)
    validation = _read_list(root / "validation_list.txt")
    testing = _read_list(root / "testing_list.txt")
    rng = np.random.default_rng(seed)

    targets: dict[str, list[Example]] = {"train": [], "val": [], "test": []}
    unknowns: dict[str, list[Example]] = {"train": [], "val": [], "test": []}

    for word_dir in sorted(p for p in root.iterdir() if p.is_dir() and p.name != NOISE_DIR):
        word = word_dir.name
        is_target = word in TARGET_WORDS
        for path in sorted(word_dir.glob("*.wav")):
            relative = f"{word}/{path.name}"
            split = _which_split(relative, validation, testing)
            bucket = targets if is_target else unknowns
            bucket[split].append(
                Example(label=word if is_target else UNKNOWN, path=path)
            )

    noise = load_noise(root)
    spec = SplitSpec()

    for split in ("train", "val", "test"):
        kept = list(targets[split])
        n_target = len(kept)
        if n_target == 0:
            continue

        # Subsample _unknown_ deterministically by filename, so the choice does
        # not move when the rest of the pipeline changes.
        n_unknown = int(round(n_target * unknown_fraction / (1.0 - unknown_fraction)))
        pool = sorted(unknowns[split], key=lambda e: _deterministic_unit(str(e.path)))
        kept.extend(pool[:n_unknown])

        n_silence = int(round(n_target * silence_fraction / (1.0 - silence_fraction)))
        for _ in range(n_silence):
            kept.append(
                Example(
                    label=SILENCE,
                    noise_index=int(rng.integers(0, max(len(noise), 1))),
                    noise_offset=int(rng.integers(0, 10**7)),
                    gain=float(rng.uniform(0.0, 1.0)),
                )
            )

        rng.shuffle(kept)
        getattr(spec, split).extend(kept)

    return spec


def label_distribution(examples: list[Example]) -> dict[str, int]:
    """How many examples of each label."""
    counts = dict.fromkeys(LABELS, 0)
    for example in examples:
        counts[example.label] += 1
    return counts


def speakers_in(examples: list[Example]) -> set[str]:
    """Speaker ids present, for checking that splits really are disjoint."""
    return {speaker_id(e.path) for e in examples if e.path is not None}
