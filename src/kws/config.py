"""Typed configuration loaded from YAML."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any

import yaml


@dataclass
class DataConfig:
    root: str = "data/speech_commands"
    unknown_fraction: float = 0.1
    silence_fraction: float = 0.1
    seed: int = 7
    cache_dir: str = "data/cache"


@dataclass
class FrontendSection:
    window_ms: float = 40.0
    hop_ms: float = 20.0
    n_mels: int = 40
    n_mfcc: int = 10
    fmin: float = 20.0
    fmax: float = 7600.0


@dataclass
class ModelConfig:
    channels: int = 64
    n_blocks: int = 4
    dropout: float = 0.2


@dataclass
class TrainConfig:
    epochs: int = 18
    batch_size: int = 128
    lr: float = 3e-3
    weight_decay: float = 1e-4
    patience: int = 5
    seed: int = 7
    augment: bool = True


@dataclass
class Config:
    data: DataConfig = field(default_factory=DataConfig)
    frontend: FrontendSection = field(default_factory=FrontendSection)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)

    @classmethod
    def load(cls, path: str | Path | None) -> Config:
        if path is None:
            return cls()
        raw: dict[str, Any] = yaml.safe_load(Path(path).read_text()) or {}
        unknown = set(raw) - {f.name for f in fields(cls)}
        if unknown:
            raise ValueError(f"unknown config section(s): {sorted(unknown)}")

        sections: dict[str, Any] = {}
        for f in fields(cls):
            section = f.default_factory()  # type: ignore[misc]
            for key, value in (raw.get(f.name) or {}).items():
                if not hasattr(section, key):
                    raise ValueError(f"unknown config key {f.name}.{key}")
                setattr(section, key, value)
            sections[f.name] = section
        return cls(**sections)

    def frontend_config(self):
        from kws.frontend import FrontendConfig

        return FrontendConfig(
            window_ms=self.frontend.window_ms,
            hop_ms=self.frontend.hop_ms,
            n_mels=self.frontend.n_mels,
            n_mfcc=self.frontend.n_mfcc,
            fmin=self.frontend.fmin,
            fmax=self.frontend.fmax,
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
