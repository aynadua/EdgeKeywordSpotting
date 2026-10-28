"""Metrics for the 12-class task, especially the false-alarm view."""

import numpy as np
import pytest

from kws.data import LABELS, SILENCE, TARGET_WORDS, UNKNOWN
from kws.evaluate import confusion_matrix, false_alarm_rate, per_class_recall, summarise

INDEX = {label: i for i, label in enumerate(LABELS)}


def test_confusion_rows_are_normalised():
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0, 1, 1, 1])
    matrix = confusion_matrix(y_true, y_pred, len(LABELS))
    assert matrix[0].sum() == pytest.approx(1.0)
    assert matrix[0, 0] == pytest.approx(0.5)


def test_absent_classes_do_not_produce_nan():
    matrix = confusion_matrix(np.array([0]), np.array([0]), len(LABELS))
    assert np.all(np.isfinite(matrix))


def test_per_class_recall_is_keyed_by_label():
    y = np.arange(len(LABELS))
    recalls = per_class_recall(y, y)
    assert set(recalls) == set(LABELS)
    assert all(v == 1.0 for v in recalls.values())


def test_false_alarm_counts_only_keyword_predictions():
    """Calling unknown audio '_silence_' is not a false wake."""
    y_true = np.array([INDEX[UNKNOWN]] * 4)
    y_pred = np.array([INDEX["yes"], INDEX[SILENCE], INDEX[UNKNOWN], INDEX["go"]])

    rates = false_alarm_rate(y_true, y_pred)
    assert rates["false_alarm_on_unknown"] == pytest.approx(0.5)


def test_silence_and_unknown_are_reported_separately():
    y_true = np.array([INDEX[SILENCE], INDEX[UNKNOWN]])
    y_pred = np.array([INDEX["yes"], INDEX[UNKNOWN]])

    rates = false_alarm_rate(y_true, y_pred)
    assert rates["false_alarm_on_silence"] == pytest.approx(1.0)
    assert rates["false_alarm_on_unknown"] == pytest.approx(0.0)


def test_a_perfect_model_never_false_alarms():
    y = np.arange(len(LABELS))
    assert false_alarm_rate(y, y)["false_alarm_overall"] == 0.0


def test_keyword_accuracy_excludes_silence_and_unknown():
    y_true = np.array([INDEX[SILENCE], INDEX[UNKNOWN], INDEX["yes"], INDEX["no"]])
    # Both keywords right, both non-keywords wrong.
    y_pred = np.array([INDEX["go"], INDEX["go"], INDEX["yes"], INDEX["no"]])

    record = summarise("m", y_true, y_pred)
    assert record["keyword_accuracy"] == pytest.approx(1.0)
    assert record["accuracy"] == pytest.approx(0.5)


def test_summary_carries_labels_and_sizes():
    y = np.arange(len(LABELS))
    record = summarise("m", y, y, n_parameters=123)
    assert record["labels"] == list(LABELS)
    assert record["n_test"] == len(LABELS)
    assert record["n_parameters"] == 123
    assert len(record["confusion_matrix"]) == len(LABELS)


def test_target_words_are_the_ten_standard_keywords():
    assert set(TARGET_WORDS) == {
        "yes", "no", "up", "down", "left", "right", "on", "off", "stop", "go",
    }
