"""Helpers compartidos por train.py y evaluate.py."""
from pathlib import Path
import numpy as np
from preprocessing.dataset_loader import load_subset
from preprocessing.splits import load_splits
from src.utils.config import load_config


def processed_paths(pre):
    d = Path("data/processed")
    return d / f"{pre.name}_subset.npz", d / f"{pre.name}_splits.npz"


def load_all(cfg):
    pre = load_config(cfg.data.preprocess_config)
    sp, spl = processed_paths(pre)
    if not sp.exists() or not spl.exists():
        raise FileNotFoundError(f"Falta {sp.name}/{spl.name}. Ejecuta primero:\n"
                                f"  python -m scripts.preprocess --config {cfg.data.preprocess_config}")
    return pre, load_subset(sp), load_splits(spl)
