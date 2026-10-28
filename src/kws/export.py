"""ONNX export, int8 quantisation and the size/latency budget.

The target is a microcontroller, so the artefact's *size* and *latency* are
results, not footnotes. The budget this repo holds itself to is the one from
the Hello Edge work: under 80 KB of weights and under 10 ms per inference, so
a 1 Hz wake-word check costs well under 1% duty cycle.

Quantisation is static and calibrated on real training features. Dynamic
quantisation would leave the convolutions in float, which is most of this
network.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np
import onnxruntime as ort
import torch
from onnxruntime.quantization import CalibrationDataReader, QuantFormat, QuantType, quantize_static
from onnxruntime.quantization.shape_inference import quant_pre_process
from torch import nn

INPUT_NAME = "mfcc"
OUTPUT_NAME = "logits"

SIZE_BUDGET_KB = 80.0
LATENCY_BUDGET_MS = 10.0


def export_onnx(model: nn.Module, n_frames: int, n_mfcc: int, path: str | Path) -> Path:
    """Write an fp32 ONNX graph with a dynamic batch dimension."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    model.eval()
    dummy = torch.randn(1, 1, n_frames, n_mfcc)
    torch.onnx.export(
        model,
        dummy,
        str(path),
        input_names=[INPUT_NAME],
        output_names=[OUTPUT_NAME],
        dynamic_axes={INPUT_NAME: {0: "batch"}, OUTPUT_NAME: {0: "batch"}},
        opset_version=17,
        dynamo=False,
    )
    return path


class _ArrayCalibrationReader(CalibrationDataReader):
    def __init__(self, x: np.ndarray) -> None:
        self._batches = iter(
            [{INPUT_NAME: example[None, None].astype(np.float32)} for example in x]
        )

    def get_next(self) -> dict[str, np.ndarray] | None:
        return next(self._batches, None)


def quantise_int8(fp32_path: str | Path, calibration_x: np.ndarray, path: str | Path) -> Path:
    """Static post-training quantisation, calibrated on real features."""
    fp32_path, path = Path(fp32_path), Path(path)
    prepared = path.with_name(path.stem + ".prepared.onnx")
    quant_pre_process(str(fp32_path), str(prepared))
    quantize_static(
        str(prepared),
        str(path),
        _ArrayCalibrationReader(calibration_x),
        quant_format=QuantFormat.QDQ,
        activation_type=QuantType.QUInt8,
        weight_type=QuantType.QInt8,
        per_channel=True,
    )
    prepared.unlink(missing_ok=True)
    return path


def session(path: str | Path) -> ort.InferenceSession:
    """Single-threaded, so latency means one core -- as on a device."""
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    return ort.InferenceSession(str(path), options, providers=["CPUExecutionProvider"])


def predict_onnx(sess: ort.InferenceSession, x: np.ndarray, batch_size: int = 256) -> np.ndarray:
    out = []
    for start in range(0, len(x), batch_size):
        batch = x[start : start + batch_size].astype(np.float32)[:, None]
        out.append(sess.run([OUTPUT_NAME], {INPUT_NAME: batch})[0])
    return np.concatenate(out).argmax(axis=1)


def benchmark(
    sess: ort.InferenceSession, n_frames: int, n_mfcc: int, n_runs: int = 300, warmup: int = 30
) -> dict[str, float]:
    """Single-example, single-thread latency in milliseconds."""
    rng = np.random.default_rng(0)
    example = rng.standard_normal((1, 1, n_frames, n_mfcc)).astype(np.float32)

    for _ in range(warmup):
        sess.run([OUTPUT_NAME], {INPUT_NAME: example})

    timings = np.empty(n_runs)
    for i in range(n_runs):
        started = time.perf_counter()
        sess.run([OUTPUT_NAME], {INPUT_NAME: example})
        timings[i] = (time.perf_counter() - started) * 1e3

    return {
        "latency_ms_mean": float(np.mean(timings)),
        "latency_ms_p50": float(np.percentile(timings, 50)),
        "latency_ms_p95": float(np.percentile(timings, 95)),
    }


def artefact_report(path: str | Path, n_frames: int, n_mfcc: int, **extra: Any) -> dict[str, Any]:
    """Size, latency, and whether the MCU budget is met."""
    path = Path(path)
    size_kb = path.stat().st_size / 1024
    timings = benchmark(session(path), n_frames, n_mfcc)
    return {
        "artefact": path.name,
        "size_kb": round(size_kb, 1),
        **timings,
        "within_size_budget": size_kb <= SIZE_BUDGET_KB,
        "within_latency_budget": timings["latency_ms_p95"] <= LATENCY_BUDGET_MS,
        **extra,
    }
