"""Particiones train/val/test por señal, ESTRATIFICADAS por transmisor.

Se calculan UNA vez sobre los índices de señales (antes de generar ninguna resolución).
Todas las resoluciones de una misma señal heredan su partición -> sin fuga de información.
"""
from pathlib import Path
import numpy as np


def stratified_split(y, train=0.70, val=0.15, test=0.15, seed=42):
    assert abs(train + val + test - 1.0) < 1e-6
    rng = np.random.default_rng(seed)
    parts = {"train": [], "val": [], "test": []}
    for c in np.unique(y):
        idx = rng.permutation(np.where(y == c)[0])
        n_tr, n_va = int(round(len(idx) * train)), int(round(len(idx) * val))
        parts["train"].append(idx[:n_tr])
        parts["val"].append(idx[n_tr:n_tr + n_va])
        parts["test"].append(idx[n_tr + n_va:])
    out = {k: np.sort(np.concatenate(v)) for k, v in parts.items()}
    assert not (set(out["train"]) & set(out["val"]) | set(out["train"]) & set(out["test"]) | set(out["val"]) & set(out["test"]))
    return out


def holdout_date_split(y, date_id, holdout_dates, val=0.15, seed=42):
    """Prueba = TODAS las señales de las fechas reservadas (el modelo nunca ve esos días).
    Entrenamiento/validación = resto de fechas, partición aleatoria estratificada por transmisor."""
    hold = np.isin(date_id, list(holdout_dates))
    if not hold.any() or hold.all():
        raise ValueError("holdout_dates debe dejar señales tanto para entrenamiento como para prueba.")
    rng = np.random.default_rng(seed)
    tr, va = [], []
    for c in np.unique(y):
        idx = rng.permutation(np.where((y == c) & ~hold)[0])
        n_va = int(round(len(idx) * val))
        va.append(idx[:n_va]); tr.append(idx[n_va:])
    out = {"train": np.sort(np.concatenate(tr)), "val": np.sort(np.concatenate(va)), "test": np.where(hold)[0]}
    assert not set(np.unique(date_id[out["test"]])) & set(np.unique(date_id[out["train"]]))
    return out


def save_splits(splits, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **splits)


def load_splits(path):
    z = np.load(path)
    return {k: z[k] for k in z.files}


if __name__ == "__main__":
    y = np.repeat(np.arange(6), 1000)
    s = stratified_split(y)
    print({k: len(v) for k, v in s.items()})
    for k, v in s.items():
        assert np.bincount(y[v]).min() > 0
    d = np.tile(np.repeat(np.arange(4), 250), 6)
    h = holdout_date_split(y, d, [3])
    assert set(d[h["test"]]) == {3} and 3 not in set(d[h["train"]]) | set(d[h["val"]])
    assert len(h["train"]) + len(h["val"]) + len(h["test"]) == len(y)
    print("holdout por fecha:", {k: len(v) for k, v in h.items()})
    print("splits OK")
