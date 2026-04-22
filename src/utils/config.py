"""Dataclass configs for training and data generation."""

from dataclasses import dataclass, field


@dataclass
class DataConfig:
    re_min: float = 20.0
    re_max: float = 200.0
    n_samples: int = 400
    nx: int = 128
    ny: int = 128
    raw_dir: str = "data/raw"
    processed_dir: str = "data/processed"
    splits_dir: str = "data/splits"
    train_ratio: float = 0.7
    val_ratio: float = 0.15


@dataclass
class TrainingConfig:
    model_type: str = "fno_baseline"
    in_channels: int = 3
    out_channels: int = 3
    fno_modes: int = 16
    fno_width: int = 64
    n_layers: int = 4
    epochs: int = 200
    batch_size: int = 16
    lr: float = 1e-3
    pde_weight: float = 0.1
    device: str = "cuda"
    checkpoint_dir: str = "models/fno_baseline"
    use_pino: bool = False
