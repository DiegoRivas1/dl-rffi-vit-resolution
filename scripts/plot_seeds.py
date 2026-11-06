"""Gráfico AGREGADO entre semillas: media ± desviación (barras) y cada semilla (puntos).

Compara ViT vs MSViT, con STFT regenerada y con redimensionamiento, para modelos entrenados en R_train.
Agrupa automáticamente las semillas (f1_vit_base_r64, ..._s43, ..._s44) por tipo de modelo.

    python -m scripts.plot_seeds                      # los 3 modos de interpolación
    python -m scripts.plot_seeds --modes bilinear
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.utils.naming import fig_name

NAME = {"vit": "ViT", "msvit": "MSViT"}
COLOR = {"vit": "#1f77b4", "msvit": "#ff7f0e"}


def collect(runs_dir, r_train, mode):
    parts = []
    for p in Path(runs_dir).glob("*/metrics/eval_by_resolution.csv"):
        d = pd.read_csv(p)
        d = d[(~d["finetune"].astype(bool)) & (d["R_train"] == r_train) & (d["phase"].isin([1, 3]))]
        parts.append(d.assign(method="STFT regenerada"))
    for p in Path(runs_dir).glob(f"*/metrics/eval_resize_{mode}.csv"):
        d = pd.read_csv(p)
        parts.append(d[d["R_train"] == r_train].assign(method=f"redimensionada ({mode})"))
    if not parts:
        raise SystemExit("No se encontraron CSV en runs/. ¿Corriste train y eval_resize?")
    return pd.concat(parts, ignore_index=True)


def make_figure(df, mode, out_dir):
    order = list(df.drop_duplicates("R_test").sort_values(["H", "W"])["R_test"])
    methods = ["STFT regenerada", f"redimensionada ({mode})"]
    series = [(m, meth) for meth in methods for m in ("vit", "msvit") if ((df["model"] == m) & (df["method"] == meth)).any()]
    n_by_model = df.groupby("model")["experiment"].nunique().to_dict()
    x, w = np.arange(len(order)), 0.8 / len(series)
    rng = np.random.default_rng(0)
    fig, ax = plt.subplots(figsize=(9, 4.4))
    for i, (m, meth) in enumerate(series):
        sub = df[(df["model"] == m) & (df["method"] == meth)]
        pos = x + (i - (len(series) - 1) / 2) * w
        means = [sub[sub["R_test"] == r]["accuracy"].mean() for r in order]
        stds = [np.nan_to_num(sub[sub["R_test"] == r]["accuracy"].std(ddof=1)) for r in order]
        hatch = "//" if meth.startswith("redim") else None
        ax.bar(pos, means, w, yerr=stds, capsize=2, color=COLOR[m], alpha=0.75, hatch=hatch, edgecolor="white",
               label=f"{NAME[m]} · {meth} (n={n_by_model[m]})")
        for j, r in enumerate(order):
            ys = sub[sub["R_test"] == r]["accuracy"].values
            ax.scatter(pos[j] + rng.uniform(-w / 4, w / 4, len(ys)), ys, s=9, color="black", zorder=3)
    ax.axhline(1 / 6, color="gray", ls="--", lw=1, label="azar (1/6)")
    ax.set_xticks(x); ax.set_xticklabels(order); ax.set_ylim(0, 1.05)
    ax.set_xlabel("Resolución de prueba"); ax.set_ylabel("Accuracy (media ± desv. entre semillas)")
    ax.grid(alpha=.3, axis="y"); ax.legend(fontsize=7, loc="upper right", framealpha=0.95)
    fig.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / fig_name("seeds", "acc_mean_std", mode=mode, n=max(n_by_model.values()))
    fig.savefig(path, dpi=150); plt.close(fig)
    summ = (df.groupby(["model", "method", "R_test", "H", "W"])["accuracy"].agg(["mean", "std", "count"]).round(4)
              .reset_index().sort_values(["method", "model", "H", "W"]))
    csv = out_dir / fig_name("seeds", "summary", mode=mode).replace(".png", ".csv")
    summ.to_csv(csv, index=False)
    print("figura:", path, "\ntabla :", csv)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-dir", default="runs")
    ap.add_argument("--r-train", default="R64x64")
    ap.add_argument("--modes", nargs="+", default=["bilinear", "bicubic", "nearest"])
    ap.add_argument("--out", default="figures/comparison")
    a = ap.parse_args()
    for mode in a.modes:
        make_figure(collect(a.runs_dir, a.r_train, mode), mode, Path(a.out))


if __name__ == "__main__":
    main()
