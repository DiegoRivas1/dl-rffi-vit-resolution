"""CONTROL: evaluación con REDIMENSIONAMIENTO (sin reentrenar).

Toma los espectrogramas de R_train (p. ej. 64x64), los reescala por interpolación al tamaño de cada
resolución de prueba y evalúa los modelos ya entrenados (best.pt). Es el régimen para el que MSViT fue
diseñado (cambio de escala de una imagen), y se contrasta con la regeneración de la STFT (eval_by_resolution.csv).

    python -m scripts.eval_resize --runs f1_vit_base_r64 f3_msvit_base_r64
    python -m scripts.eval_resize --runs ... --mode bicubic        # bilinear (def.) | bicubic | nearest
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import TensorDataset

from src.evaluation.eval_resolutions import predict
from src.evaluation.metrics import compute_metrics, delta
from src.models import build_model
from src.training.checkpoint import load_checkpoint
from src.training.data import build_dataset, make_loader, res_hw
from src.training.setup import load_all
from src.utils.config import load_config
from src.utils.naming import fig_name


def resize_batch(X, hw, mode, renorm):
    kw = {} if mode == "nearest" else {"align_corners": False}
    if mode != "nearest" and (hw[0] < X.shape[2] or hw[1] < X.shape[3]):
        kw["antialias"] = True                      # al reducir, sin aliasing (comportamiento estándar)
    Y = F.interpolate(X, size=hw, mode=mode, **kw)
    if renorm:                                      # misma normalización por muestra que el pipeline
        m = Y.mean(dim=(1, 2, 3), keepdim=True)
        s = Y.std(dim=(1, 2, 3), keepdim=True)
        Y = (Y - m) / (s + 1e-6)
    return Y


def eval_run(run, runs_dir, mode, renorm, device):
    run_dir = Path(runs_dir) / run
    cfg = load_config(run_dir / "config_snapshot.yaml")
    pre, subset, splits = load_all(cfg)
    n_cls = len(subset["tx_names"])
    r_tr = cfg.resolution.train
    hw_tr = res_hw(pre.resolutions[r_tr], pre.stft.signal_len)
    model = build_model(cfg.model, n_cls, hw_tr)
    ck = load_checkpoint(run_dir / "checkpoints" / "best.pt", device)
    model.load_state_dict(ck["model"]); model.to(device).eval()
    X, Y = build_dataset(subset, splits["test"], pre.resolutions[r_tr], pre.stft, device).tensors
    rows, base = [], None
    order = [r_tr] + [r for r in cfg.resolution.test if r != r_tr]
    for rname in order:
        hw = res_hw(pre.resolutions[rname], pre.stft.signal_len)
        Xr = X if tuple(hw) == tuple(X.shape[2:]) else resize_batch(X, hw, mode, renorm)
        yt, yp = predict(model, make_loader(TensorDataset(Xr, Y), 256, False), device)
        m = compute_metrics(yt, yp, n_cls)
        base = m["accuracy"] if rname == r_tr else base
        d, dp = delta(base, m["accuracy"])
        rows.append({"experiment": run, "model": cfg.model.type, "eval_type": f"resize-{mode}",
                     "finetune": False, "R_train": r_tr, "R_test": rname, "H": hw[0], "W": hw[1],
                     "accuracy": m["accuracy"], "precision_macro": m["precision_macro"],
                     "recall_macro": m["recall_macro"], "f1_macro": m["f1_macro"],
                     "delta_acc": d, "delta_acc_pct": dp, "seed": cfg.experiment.seed})
        print(f"  {run:>22} | {rname:>9} ({hw[0]}x{hw[1]}): acc={m['accuracy']:.4f} f1={m['f1_macro']:.4f}")
    df = pd.DataFrame(rows)
    (run_dir / "metrics").mkdir(parents=True, exist_ok=True)
    df.to_csv(run_dir / "metrics" / f"eval_resize_{mode}.csv", index=False)
    return df


def compare_figure(runs, runs_dir, resize_dfs, mode, out_dir):
    series, labels = [], []
    for run in runs:
        regen = pd.read_csv(Path(runs_dir) / run / "metrics" / "eval_by_resolution.csv")
        series += [regen.set_index("R_test")["accuracy"], resize_dfs[run].set_index("R_test")["accuracy"]]
        labels += [f"{run} (STFT regenerada)", f"{run} (redimensionada)"]
    order = list(resize_dfs[runs[0]].sort_values(["H", "W"])["R_test"])
    x, w = np.arange(len(order)), 0.8 / len(series)
    fig, ax = plt.subplots(figsize=(8, 4))
    for i, (s, lab) in enumerate(zip(series, labels)):
        ax.bar(x + (i - (len(series) - 1) / 2) * w, [s.get(r, np.nan) for r in order], w, label=lab)
    ax.axhline(1 / 6, color="gray", ls="--", lw=1, label="azar (1/6)")
    ax.set_xticks(x); ax.set_xticklabels(order); ax.set_ylim(0, 1.05); ax.set_ylabel("Accuracy")
    ax.set_xlabel("Resolución de prueba"); ax.legend(fontsize=7, ncol=2); ax.grid(alpha=.3, axis="y")
    fig.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / fig_name("control_resize", "acc_by_resolution", mode=mode, n_models=len(runs))
    fig.savefig(path, dpi=150); plt.close(fig)
    print("figura:", path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--runs-dir", default="runs")
    ap.add_argument("--mode", default="bilinear", choices=["bilinear", "bicubic", "nearest"])
    ap.add_argument("--no-renorm", action="store_true", help="no re-estandarizar tras reescalar")
    ap.add_argument("--device", default=None)
    a = ap.parse_args()
    device = torch.device(a.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    dfs = {r: eval_run(r, a.runs_dir, a.mode, not a.no_renorm, device) for r in a.runs}
    compare_figure(a.runs, a.runs_dir, dfs, a.mode, Path("figures/comparison"))


if __name__ == "__main__":
    main()
