"""BARRIDO DE VENTANA: misma imagen (F x T de R_train), misma rejilla de parches; solo cambia la longitud de la ventana STFT.

Aísla el efecto del compromiso tiempo-frecuencia: n_fft (nº de bins) y hop (nº de frames) son los de R_train, así que
no cambia ni un píxel de la forma de entrada ni hace falta interpolar o escalar nada. Sin reentrenar (usa best.pt).

    python -m scripts.eval_window_sweep --runs f1_vit_base_r64 f1_vit_base_r64_s43 ... f3_msvit_base_r64 f3_msvit_base_r64_s43 ...
    python -m scripts.eval_window_sweep --runs ... --windows 32 48 56 64 72 96 128
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import torch

from src.evaluation.eval_resolutions import predict
from src.evaluation.metrics import compute_metrics
from src.models import build_model
from src.training.checkpoint import load_checkpoint
from src.training.data import build_dataset, make_loader, res_hw
from src.training.setup import load_all
from src.utils.config import Cfg, load_config
from src.utils.naming import fig_name

NAME = {"vit": "ViT", "msvit": "MSViT"}
COLOR = {"vit": "#1f77b4", "msvit": "#ff7f0e"}


def eval_run(run, runs_dir, windows, device):
    run_dir = Path(runs_dir) / run
    cfg = load_config(run_dir / "config_snapshot.yaml")
    pre, subset, splits = load_all(cfg)
    n_cls = len(subset["tx_names"])
    r_tr = cfg.resolution.train
    res_tr = pre.resolutions[r_tr]
    model = build_model(cfg.model, n_cls, tuple(res_hw(res_tr, pre.stft.signal_len)))
    model.load_state_dict(load_checkpoint(run_dir / "checkpoints" / "best.pt", device)["model"])
    model.to(device).eval()
    rows = []
    for w in windows:
        res = Cfg(n_fft=res_tr.n_fft, win_len=w, hop=res_tr.hop)
        ds = build_dataset(subset, splits["test"], res, pre.stft, device)
        yt, yp = predict(model, make_loader(ds, 256, False), device)
        m = compute_metrics(yt, yp, n_cls)
        rows.append({"experiment": run, "model": cfg.model.type, "seed": cfg.experiment.seed, "R_train": r_tr,
                     "win_train": res_tr.win_len, "win_len": w, "accuracy": m["accuracy"], "f1_macro": m["f1_macro"]})
        print(f"  {run:>24} | ventana {w:>3} (entrenamiento {res_tr.win_len}): acc={m['accuracy']:.4f}")
    df = pd.DataFrame(rows)
    (run_dir / "metrics").mkdir(parents=True, exist_ok=True)
    df.to_csv(run_dir / "metrics" / "eval_window_sweep.csv", index=False)
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--runs-dir", default="runs")
    ap.add_argument("--windows", nargs="+", type=int, default=[32, 40, 48, 56, 60, 64, 68, 72, 80, 96, 128])
    ap.add_argument("--device", default=None)
    a = ap.parse_args()
    device = torch.device(a.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    df = pd.concat([eval_run(r, a.runs_dir, a.windows, device) for r in a.runs], ignore_index=True)
    n = df["experiment"].nunique()
    summ = df.groupby(["model", "win_len"])["accuracy"].agg(["mean", "std", "count"]).round(4).reset_index()
    out = Path("figures/comparison"); out.mkdir(parents=True, exist_ok=True)
    csv = out / fig_name("window_sweep", "summary", n_runs=n).replace(".png", ".csv")
    summ.to_csv(csv, index=False)

    fig, ax = plt.subplots(figsize=(8, 4.2))
    for mtype, g in summ.groupby("model"):
        ax.errorbar(g["win_len"], g["mean"], yerr=g["std"].fillna(0), marker="o", capsize=3, color=COLOR.get(mtype),
                    label=f"{NAME.get(mtype, mtype)} (n={int(g['count'].max())})")
        pts = df[df["model"] == mtype]
        ax.scatter(pts["win_len"], pts["accuracy"], s=8, color=COLOR.get(mtype), alpha=0.35, zorder=1)
    ax.axvline(int(df["win_train"].iloc[0]), color="gray", ls=":", lw=1, label="ventana de entrenamiento")
    ax.axhline(1 / 6, color="gray", ls="--", lw=1, label="azar (1/6)")
    ax.set_xscale("log", base=2); ax.set_xticks(a.windows); ax.set_xticklabels(a.windows, fontsize=8); ax.minorticks_off()
    ax.set_ylim(0, 1.05); ax.set_xlabel("Longitud de la ventana STFT (muestras); imagen siempre " + df["R_train"].iloc[0][1:])
    ax.set_ylabel("Accuracy (media ± desv. entre semillas)"); ax.grid(alpha=.3); ax.legend(fontsize=8)
    fig.tight_layout()
    png = out / fig_name("window_sweep", "acc_vs_window", n_runs=n)
    fig.savefig(png, dpi=150); plt.close(fig)
    wide = summ.assign(v=summ["mean"].map("{:.3f}".format) + " ± " + summ["std"].fillna(0).map("{:.3f}".format))
    print("\nRESUMEN (media ± desv. entre runs):\n", wide.pivot(index="win_len", columns="model", values="v").to_string(),
          "\n\ntabla :", csv, "\nfigura:", png)


if __name__ == "__main__":
    main()
