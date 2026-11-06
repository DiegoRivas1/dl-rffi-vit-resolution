"""MAPA DE SENSIBILIDAD: redimensiona el espectrograma de R_train a (F, T) arbitrarios, por separado en cada eje.

Responde: ¿el modelo tolera cambios en UN solo eje? ¿qué factor de escala? ¿frecuencia o tiempo?
Sin reentrenar: usa best.pt de cada run. Eje vertical = frecuencia (F), horizontal = tiempo (T).

    python -m scripts.eval_resize_grid --runs f1_vit_base_r64 f1_vit_base_r64_s43 f3_msvit_base_r64 f3_msvit_base_r64_s43
    python -m scripts.eval_resize_grid --runs ... --mode bilinear --sizes 32 48 64 80 96 128
"""
import argparse
import itertools
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.data import TensorDataset

from scripts.eval_resize import resize_batch
from src.evaluation.eval_resolutions import predict
from src.models import build_model
from src.training.checkpoint import load_checkpoint
from src.training.data import build_dataset, make_loader, res_hw
from src.training.setup import load_all
from src.utils.config import load_config
from src.utils.naming import fig_name


def grid_run(run, runs_dir, sizes, mode, device):
    run_dir = Path(runs_dir) / run
    cfg = load_config(run_dir / "config_snapshot.yaml")
    pre, subset, splits = load_all(cfg)
    r_tr = cfg.resolution.train
    hw_tr = res_hw(pre.resolutions[r_tr], pre.stft.signal_len)
    model = build_model(cfg.model, len(subset["tx_names"]), hw_tr)
    model.load_state_dict(load_checkpoint(run_dir / "checkpoints" / "best.pt", device)["model"])
    model.to(device).eval()
    X, Y = build_dataset(subset, splits["test"], pre.resolutions[r_tr], pre.stft, device).tensors
    rows = []
    for H, W in itertools.product(sizes, sizes):
        Xr = X if (H, W) == tuple(X.shape[2:]) else resize_batch(X, (H, W), mode, True)
        yt, yp = predict(model, make_loader(TensorDataset(Xr, Y), 256, False), device)
        rows.append({"experiment": run, "model": cfg.model.type, "eval_type": f"resize-grid-{mode}",
                     "R_train": r_tr, "H": H, "W": W, "scale_F": H / hw_tr[0], "scale_T": W / hw_tr[1],
                     "accuracy": float((yt == yp).mean()), "seed": cfg.experiment.seed})
    df = pd.DataFrame(rows)
    df.to_csv(run_dir / "metrics" / f"eval_resize_grid_{mode}.csv", index=False)
    print(f"  {run}: {len(df)} combinaciones")
    return df


def heatmaps(df, sizes, mode, out_dir):
    models = [m for m in ("vit", "msvit") if m in set(df["model"])]
    mean = {m: df[df["model"] == m].pivot_table(index="H", columns="W", values="accuracy", aggfunc="mean").loc[sizes, sizes] for m in models}
    panels = [(m, mean[m], {"vit": "ViT", "msvit": "MSViT"}[m], "viridis", (0, 1)) for m in models]
    if len(models) == 2:
        panels.append(("diff", mean["msvit"] - mean["vit"], "MSViT − ViT", "RdBu", (-0.5, 0.5)))
    fig, axes = plt.subplots(1, len(panels), figsize=(4.6 * len(panels), 4.3), squeeze=False)
    for ax, (_, M, title, cmap, (vmin, vmax)) in zip(axes[0], panels):
        im = ax.imshow(M.values, origin="lower", cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_xticks(range(len(sizes))); ax.set_xticklabels(sizes); ax.set_yticks(range(len(sizes))); ax.set_yticklabels(sizes)
        ax.set_xlabel("T (tiempo, ancho)"); ax.set_ylabel("F (frecuencia, alto)")
        n = {m: df[df["model"] == m]["experiment"].nunique() for m in models}
        ax.set_title(title + ("" if title.startswith("MSViT −") else f" (n={n[_] if _ in n else ''})"), fontsize=9)
        for i in range(len(sizes)):
            for j in range(len(sizes)):
                ax.text(j, i, f"{M.values[i, j]:.2f}", ha="center", va="center", fontsize=7, color="white" if cmap == "viridis" and M.values[i, j] < .6 else "black")
        fig.colorbar(im, ax=ax, fraction=0.046)
    fig.suptitle(f"Sensibilidad al redimensionamiento por eje ({mode}); modelos entrenados en 64×64", fontsize=9)
    fig.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / fig_name("grid", "resize_sensitivity", mode=mode, n_runs=df["experiment"].nunique())
    fig.savefig(path, dpi=150); plt.close(fig)
    print("figura:", path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--runs-dir", default="runs")
    ap.add_argument("--mode", default="bilinear", choices=["bilinear", "bicubic", "nearest"])
    ap.add_argument("--sizes", nargs="+", type=int, default=[32, 48, 64, 80, 96, 128])
    ap.add_argument("--device", default=None)
    a = ap.parse_args()
    device = torch.device(a.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    df = pd.concat([grid_run(r, a.runs_dir, a.sizes, a.mode, device) for r in a.runs], ignore_index=True)
    heatmaps(df, a.sizes, a.mode, Path("figures/comparison"))


if __name__ == "__main__":
    main()
