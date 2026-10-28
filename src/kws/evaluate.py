"""Evaluation for the 12-class task.

Overall accuracy is reported, but two breakdowns matter more for a wake word:

* **Per-class recall**, because the cost of missing "stop" is not the cost of
  missing "go", and an average hides a class that is failing.
* **False-alarm behaviour on ``_unknown_`` and ``_silence_``**. A spotter
  running continuously spends almost all of its time hearing neither a keyword
  nor nothing in particular. How often it wakes on those is the number that
  decides whether the device is usable, and it is invisible in a balanced
  accuracy figure.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from kws.data import LABELS, SILENCE, TARGET_WORDS, UNKNOWN


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> np.ndarray:
    cm = np.zeros((n_classes, n_classes), dtype=np.float64)
    np.add.at(cm, (y_true, y_pred), 1.0)
    totals = cm.sum(axis=1, keepdims=True)
    return np.divide(cm, totals, out=np.zeros_like(cm), where=totals > 0)


def per_class_recall(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    matrix = confusion_matrix(y_true, y_pred, len(LABELS))
    return {label: float(matrix[i, i]) for i, label in enumerate(LABELS)}


def false_alarm_rate(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """How often non-keyword audio is reported as a keyword."""
    keyword_indices = {LABELS.index(w) for w in TARGET_WORDS}
    out: dict[str, float] = {}

    for label in (SILENCE, UNKNOWN):
        index = LABELS.index(label)
        mask = y_true == index
        if mask.any():
            fired = np.isin(y_pred[mask], list(keyword_indices))
            out[f"false_alarm_on_{label.strip('_')}"] = float(fired.mean())

    non_keyword = ~np.isin(y_true, list(keyword_indices))
    if non_keyword.any():
        out["false_alarm_overall"] = float(
            np.isin(y_pred[non_keyword], list(keyword_indices)).mean()
        )
    return out


def summarise(name: str, y_true: np.ndarray, y_pred: np.ndarray, **extra: Any) -> dict[str, Any]:
    """The metrics record written to ``reports/``."""
    keyword_indices = [LABELS.index(w) for w in TARGET_WORDS]
    keyword_mask = np.isin(y_true, keyword_indices)

    return {
        "model": name,
        "accuracy": float(np.mean(y_true == y_pred)),
        "keyword_accuracy": float(np.mean(y_pred[keyword_mask] == y_true[keyword_mask])),
        "per_class_recall": per_class_recall(y_true, y_pred),
        **false_alarm_rate(y_true, y_pred),
        "confusion_matrix": confusion_matrix(y_true, y_pred, len(LABELS)).tolist(),
        "labels": list(LABELS),
        "n_test": int(len(y_true)),
        **extra,
    }


def write_report(record: dict[str, Any], path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=2))
    return path
