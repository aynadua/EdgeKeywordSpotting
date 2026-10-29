"""Command line entry point: ``kws <stage>``.

    kws data                  # download the corpus and cache MFCC features
    kws train --arch dscnn    # fit the DS-CNN
    kws train --arch linear   # fit the linear baseline
    kws evaluate              # score every trained model on the test split
    kws export                # ONNX export, int8 quantisation, size/latency budget
    kws report                # collect metrics and figures into reports/
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from kws import evaluate as evaluate_module
from kws import plots
from kws.config import Config
from kws.data import LABELS, build_splits, download, label_distribution, speakers_in
from kws.dataset import FeatureSet, featurise, normalise

REPORTS = Path("reports")
MODELS = Path("models")
ARCHITECTURES = ("dscnn", "linear")


def _cache(cfg: Config, split: str) -> Path:
    return Path(cfg.data.cache_dir) / f"{split}.npz"


def _load_cached(cfg: Config, split: str) -> FeatureSet:
    path = _cache(cfg, split)
    if not path.exists():
        raise SystemExit(f"missing {path}; run `kws data` first")
    return FeatureSet.load(path)


def _splits(cfg: Config):
    return build_splits(
        cfg.data.root,
        unknown_fraction=cfg.data.unknown_fraction,
        silence_fraction=cfg.data.silence_fraction,
        seed=cfg.data.seed,
    )


def cmd_data(cfg: Config, args: argparse.Namespace) -> None:
    download(cfg.data.root)
    spec = _splits(cfg)
    frontend = cfg.frontend_config()

    stats: dict[str, Any] = {"counts": spec.counts(), "labels": list(LABELS)}
    for split in ("train", "val", "test"):
        examples = getattr(spec, split)
        features = featurise(examples, cfg.data.root, frontend)
        features.save(_cache(cfg, split))
        stats[f"{split}_distribution"] = label_distribution(examples)
        print(f"{split:6s} {len(features):6d} examples  features {features.x.shape}")

    # The headline integrity property, asserted rather than assumed.
    speakers = {s: speakers_in(getattr(spec, s)) for s in ("train", "val", "test")}
    overlaps = {
        "train_val": len(speakers["train"] & speakers["val"]),
        "train_test": len(speakers["train"] & speakers["test"]),
        "val_test": len(speakers["val"] & speakers["test"]),
    }
    stats["speaker_counts"] = {k: len(v) for k, v in speakers.items()}
    stats["speaker_overlaps"] = overlaps
    print(f"speakers: { {k: len(v) for k, v in speakers.items()} }  overlaps: {overlaps}")
    if any(overlaps.values()):
        raise SystemExit("speaker leakage between splits; refusing to continue")

    REPORTS.mkdir(parents=True, exist_ok=True)
    (REPORTS / "dataset.json").write_text(json.dumps(stats, indent=2))


def cmd_train(cfg: Config, args: argparse.Namespace) -> None:
    from kws import train as train_module

    train_features = _load_cached(cfg, "train")
    val_features = _load_cached(cfg, "val")
    normalisation, (train_scaled, val_scaled) = normalise(train_features, val_features)

    spec = _splits(cfg)
    result = train_module.train(
        spec.train, train_scaled, val_scaled, cfg, normalisation, architecture=args.arch
    )
    path = train_module.save_checkpoint(result, cfg, args.arch, MODELS / f"{args.arch}.pt")
    print(
        f"best val accuracy {result.best_val_accuracy:.4f} "
        f"({result.train_seconds:.0f}s) -> {path}"
    )


def cmd_evaluate(cfg: Config, args: argparse.Namespace) -> None:
    from kws import train as train_module
    from kws.models import count_parameters

    train_features = _load_cached(cfg, "train")
    test_features = _load_cached(cfg, "test")

    for architecture in ARCHITECTURES:
        checkpoint = MODELS / f"{architecture}.pt"
        if not checkpoint.exists():
            print(f"skipping {architecture}: no checkpoint")
            continue

        model, blob = train_module.load_checkpoint(checkpoint)
        mean, std = blob["normalisation"]
        scaled = (test_features.x - mean) / std

        predictions = train_module.predict(model, scaled)
        record = evaluate_module.summarise(
            architecture, test_features.y, predictions,
            n_parameters=count_parameters(model),
            best_val_accuracy=blob["best_val_accuracy"],
            history=blob["history"],
        )
        evaluate_module.write_report(record, REPORTS / f"{architecture}.json")
        print(
            f"{architecture:8s} accuracy {record['accuracy']:.4f}  "
            f"keyword {record['keyword_accuracy']:.4f}  "
            f"false alarms {record.get('false_alarm_overall', float('nan')):.4f}"
        )
    _ = train_features


def cmd_export(cfg: Config, args: argparse.Namespace) -> None:
    from kws import export as export_module
    from kws import train as train_module

    checkpoint = MODELS / "dscnn.pt"
    if not checkpoint.exists():
        raise SystemExit("missing models/dscnn.pt; run `kws train` first")

    model, blob = train_module.load_checkpoint(checkpoint)
    mean, std = blob["normalisation"]
    train_features = _load_cached(cfg, "train")
    test_features = _load_cached(cfg, "test")
    scaled_test = (test_features.x - mean) / std
    scaled_train = (train_features.x - mean) / std

    n_frames, n_mfcc = test_features.x.shape[1], test_features.x.shape[2]
    fp32 = export_module.export_onnx(model, n_frames, n_mfcc, MODELS / "dscnn.onnx")
    int8 = export_module.quantise_int8(
        fp32, scaled_train[: args.calibration_size], MODELS / "dscnn.int8.onnx"
    )

    records: list[dict[str, Any]] = []
    for name, path in (("DS-CNN (fp32 ONNX)", fp32), ("DS-CNN (int8 ONNX)", int8)):
        predictions = export_module.predict_onnx(export_module.session(path), scaled_test)
        summary = evaluate_module.summarise(name, test_features.y, predictions)
        summary.update(export_module.artefact_report(path, n_frames, n_mfcc))
        records.append(summary)
        within_budget = summary["within_size_budget"] and summary["within_latency_budget"]
        print(
            f"{name:20s} {summary['size_kb']:7.1f} KB  "
            f"p50 {summary['latency_ms_p50']:.3f} ms  "
            f"acc {summary['accuracy']:.4f}  "
            f"budget {'PASS' if within_budget else 'FAIL'}"
        )

    evaluate_module.write_report({"artefacts": records}, REPORTS / "export.json")


def cmd_report(cfg: Config, args: argparse.Namespace) -> None:
    records: list[dict[str, Any]] = []
    for architecture in ARCHITECTURES:
        path = REPORTS / f"{architecture}.json"
        if path.exists():
            records.append(json.loads(path.read_text()))

    export_path = REPORTS / "export.json"
    artefacts = json.loads(export_path.read_text())["artefacts"] if export_path.exists() else []
    if not records:
        raise SystemExit("nothing to report; run `kws train` and `kws evaluate` first")

    plots.features(REPORTS / "frontend.png", cfg.data.root)
    plots.budget(records + artefacts, REPORTS / "budget.png")
    for record in records:
        plots.confusion(record, REPORTS / f"confusion_{record['model']}.png")
        if record.get("history"):
            plots.training_curve(record["history"], REPORTS / f"training_{record['model']}.png")

    lines = [
        "# Results",
        "",
        "Google Speech Commands V2, 12-class task "
        "(10 keywords + `_silence_` + `_unknown_`), official speaker-disjoint splits.",
        "",
        "| Model | Accuracy | Keyword accuracy | False alarms | Params | Size | p50 latency |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for record in records + artefacts:
        size = f"{record['size_kb']:.0f} KB" if "size_kb" in record else "n/a"
        latency = f"{record['latency_ms_p50']:.2f} ms" if "latency_ms_p50" in record else "n/a"
        params = f"{record['n_parameters']:,}" if "n_parameters" in record else "n/a"
        alarms = record.get("false_alarm_overall", float("nan"))
        lines.append(
            f"| {record['model']} | {record['accuracy']:.4f} | "
            f"{record['keyword_accuracy']:.4f} | {alarms:.4f} | "
            f"{params} | {size} | {latency} |"
        )

    best = max(records, key=lambda r: r["accuracy"])
    lines += [
        "",
        f"## Per-class recall ({best['model']})",
        "",
        "| Class | Recall |",
        "|---|---:|",
    ]
    for label, value in best["per_class_recall"].items():
        lines.append(f"| `{label}` | {value:.3f} |")
    lines += ["", "![budget](budget.png)", ""]

    out = REPORTS / "RESULTS.md"
    out.write_text("\n".join(lines))
    print(f"wrote {out}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="kws", description=__doc__)
    parser.add_argument("--config", default="configs/default.yaml")
    sub = parser.add_subparsers(dest="stage", required=True)

    for name, handler in (
        ("data", cmd_data),
        ("evaluate", cmd_evaluate),
        ("report", cmd_report),
    ):
        sub.add_parser(name, help=name).set_defaults(handler=handler)

    train_parser = sub.add_parser("train", help="fit a model")
    train_parser.add_argument("--arch", choices=ARCHITECTURES, default="dscnn")
    train_parser.set_defaults(handler=cmd_train)

    export_parser = sub.add_parser("export", help="ONNX export and int8 quantisation")
    export_parser.add_argument("--calibration-size", type=int, default=400)
    export_parser.set_defaults(handler=cmd_export)

    args = parser.parse_args(argv)
    cfg = Config.load(args.config) if Path(args.config).exists() else Config()
    args.handler(cfg, args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
