# Edge keyword spotting

An always-on wake-word spotter for Google Speech Commands V2, built to a
microcontroller budget: an MFCC front end written from the FFT up so it stays
implementable in C, a depthwise-separable CNN of ~23k parameters, and int8
quantisation with the size and latency cost measured rather than assumed.

> **Status:** the full benchmark run is reproduced by `make all`; measured
> results are committed in `reports/` and summarised below.

---

## The task, and why it is the 12-class one

The benchmark everybody quotes is **V2-12**: ten target keywords — *yes, no,
up, down, left, right, on, off, stop, go* — plus `_silence_` and `_unknown_`.

That framing is not decoration. A wake-word system spends essentially all of
its life hearing neither a keyword nor nothing in particular: it hears the
twenty-five other words in the corpus, and it hears a room. How often it wakes
on those decides whether the device is usable, and a 35-way word-classification
accuracy skips both halves of that job. So `_unknown_` and `_silence_` are
classes here, and the false-alarm rate on them is reported separately from
accuracy.

## Splits: the defect that inflates this benchmark

Speech Commands ships `validation_list.txt` and `testing_list.txt`, which
partition the corpus by a **hash of the speaker id**. Using them is not
optional. Splitting at random instead lets the same voice saying the same word
land in both train and test, and measures speaker memorisation rather than
keyword recognition.

The pipeline uses the official lists and then **asserts** the property rather
than trusting it:

```
train   37607 examples  features (37607, 49, 10)
val      4525 examples  features (4525, 49, 10)
test     4980 examples  features (4980, 49, 10)
speakers: {'train': 2023, 'val': 246, 'test': 241}
overlaps: {'train_val': 0, 'train_test': 0, 'val_test': 0}
```

`kws data` exits non-zero if any speaker appears in two splits.

## The front end, written out

```
1 s of 16 kHz audio
  → pre-emphasis (0.97)
  → 40 ms frames, 20 ms hop, Hann window        → 49 frames
  → 1024-point rFFT, power spectrum             → 513 bins
  → 40 triangular mel filters, 20 Hz – 7.6 kHz  → 40 bands
  → log
  → orthonormal DCT-II, keep 10 coefficients    → 10 × 49
```

490 values per inference, under 2 KB in int8 — a sane working set for a device
with 64 KB of RAM.

It is written from the FFT up rather than calling `librosa` because an
always-on spotter runs its feature extraction *on the microcontroller*,
continuously, on a battery. Whatever happens in the front end has to be a few
hundred lines of C against a CMSIS-DSP FFT, and `librosa.feature.mfcc` hides
exactly the decisions — frame size, FFT length, filterbank layout, how many
coefficients survive — that determine whether that is possible.

Being hand-written, it is verified against physics rather than against another
library: a 250 Hz, 1 kHz or 3 kHz tone must peak in the mel band that contains
it; digital silence must not produce `-inf`; the geometry must yield exactly
49 frames.

![front end](reports/frontend.png)

### A limitation the tests pin down

Scaling a waveform multiplies every mel band's power by the same factor, which
after the log is a constant offset, which lands entirely in C0. That holds
*when every band carries real energy* — verified on broadband noise, where
halving the gain moves C0 by 8.8 and the remaining coefficients by 0.005.

It does **not** hold for speech. Speech leaves many mel bands near-empty, and
in those the log is dominated by the floor, which does not scale. So halving a
real clip's gain changes the *shape* of the feature vector, not just its
level, and the change leaks into C1 and above. The practical consequence: this
front end is not level-invariant, and a model trained at one recording gain
degrades at another. Per-utterance mean normalisation or AGC ahead of the
front end is the usual fix and is **not** implemented here. Both behaviours
are asserted in `tests/test_frontend.py` so neither can regress silently.

## The model

**DS-CNN**, the reference microcontroller architecture from ARM's *Hello Edge*
work. A wide first convolution spanning the MFCC axis, then four
depthwise-separable blocks, global average pooling, a linear head.

The arithmetic is the whole point: a full 3×3 convolution over C channels
costs `9·C·C` multiplies per position, a depthwise 3×3 followed by a 1×1
pointwise costs `9·C + C·C`. At 64 channels that is 36,864 against 4,672 —
roughly eight times cheaper, for a small accuracy cost. On a device whose
inference budget is measured in milliwatts, that trade is obviously worth
making.

**The baseline is a linear model** on time-pooled MFCC statistics (mean and
standard deviation per coefficient, 252 parameters). It is there so the CNN
has to justify itself. It is also permutation-invariant over time by
construction, which makes the comparison informative: whatever the CNN gains,
it gains from temporal structure.

**Augmentation** is random time shift (±100 ms) and background noise mixing,
re-applied every epoch to the *waveform* — so the front end sees a different
signal each time — rather than cached once. A wake word never starts exactly
at the window boundary, and a device listening in a kitchen hears a kitchen.

## Deployment budget

The budget is the Hello Edge one: **under 80 KB** of weights and **under
10 ms** per inference on one core, so a 1 Hz wake check costs well under 1%
duty cycle. `kws export` reports `PASS`/`FAIL` against it rather than leaving
the reader to judge.

Quantisation is **static** post-training int8, calibrated on real training
features. Dynamic quantisation would leave the convolutions in float, which is
most of this network.

![budget](reports/budget.png)

## Reproducing

```bash
pip install -e ".[dev]"
```

```bash
make all
```

Downloads Speech Commands V2 (~2.3 GB), caches MFCC features, trains the
baseline and the DS-CNN, evaluates on the test split, exports and quantises,
and writes `reports/`. Around 30 minutes on 4 CPU cores. No GPU.

Stages run independently:

```bash
kws data && kws train --arch dscnn && kws evaluate && kws export && kws report
```

```bash
pytest
```

## Honest limits

- **Accuracy is measured on 1-second pre-segmented clips**, which is not how a
  wake word is used. A deployed spotter runs a sliding window over continuous
  audio, and its real figure of merit is false alarms per hour against a
  detection threshold. Streaming evaluation is not implemented.
- **No on-device measurement.** Latency here is single-threaded x86 through
  ONNX Runtime, which is a proxy for an MCU, not a substitute. Real numbers
  need CMSIS-NN on the target part.
- **int8 only.** Whether this survives further compression — structured
  pruning, 4-bit weights — is unexplored.
- **English, and the corpus's speaker distribution.** Speech Commands skews
  towards certain accents; per-accent performance is unmeasured and is the
  kind of gap that matters for a consumer device.
- **No level normalisation**, as described above.

## Layout

```
src/kws/
  frontend.py   framing, mel filterbank, MFCC -- written from the FFT up
  data.py       12-class task assembly, official speaker-disjoint splits
  dataset.py    feature caching, augmentation, train-only normalisation
  models.py     DS-CNN and the linear baseline
  train.py      training loop with per-epoch re-augmentation
  evaluate.py   accuracy, per-class recall, false-alarm rates
  export.py     ONNX export, static int8 quantisation, MCU budget check
  plots.py      figures
  cli.py        `kws <stage>`
tests/          59 tests, including the speaker-leakage and front-end checks
```

## References

- Warden, *Speech Commands: a dataset for limited-vocabulary speech
  recognition*, 2018.
- Zhang, Suda, Lai & Chandra, *Hello Edge: Keyword Spotting on
  Microcontrollers*, 2017.
- Howard et al., *MobileNets: efficient convolutional neural networks for
  mobile vision applications*, 2017 — for depthwise separable convolutions.

## Licence

MIT.
