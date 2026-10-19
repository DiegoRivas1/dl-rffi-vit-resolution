"""Evaluación SIN fine-tuning: pesos fijos, solo cambia la resolución de la entrada."""
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.evaluation.metrics import compute_metrics, delta
from src.training.data import build_dataset, make_loader, res_hw
from src.utils.naming import fig_name


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    ys, ps = [], []
    for x, y in loader:
        ps.append(model(x.to(device)).argmax(1).cpu()); ys.append(y)
    return torch.cat(ys).numpy(), torch.cat(ps).numpy()


def _confmat_fig(cm, names, title, path):
    fig, ax = plt.subplots(figsize=(5, 4.5))
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(names))); ax.set_xticklabels(names, rotation=45, ha="right", fontsize=7)
    ax.set_yticks(range(len(names))); ax.set_yticklabels(names, fontsize=7)
    for i in range(len(names)):
        for j in range(len(names)):
            ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=7, color="black" if cm[i, j] < cm.max() / 2 else "white")
    ax.set_xlabel("Predicho"); ax.set_ylabel("Real"); ax.set_title(title, fontsize=9)
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def run_eval(cfg, pre, subset, splits, model, device, run_dir, tag="test"):
    from pathlib import Path
    run_dir = Path(run_dir)
    (run_dir / "metrics").mkdir(parents=True, exist_ok=True)
    (run_dir / "figures").mkdir(parents=True, exist_ok=True)
    exp, names = cfg.experiment.name, [str(t) for t in subset["tx_names"]]
    order = [cfg.resolution.train] + [r for r in cfg.resolution.test if r != cfg.resolution.train]
    rows, base = [], None
    for rname in order:
        res = pre.resolutions[rname]
        ds = build_dataset(subset, splits["test"], res, pre.stft, device)
        yt, yp = predict(model, make_loader(ds, 256, False), device)
        m = compute_metrics(yt, yp, len(names))
        H, W = res_hw(res, pre.stft.signal_len)
        if rname == cfg.resolution.train:
            base = m["accuracy"]
        d, dp = delta(base, m["accuracy"])
        rows.append({"experiment": exp, "phase": cfg.experiment.phase, "model": cfg.model.type,
                     "finetune": False, "R_train": cfg.resolution.train, "R_test": rname, "H": H, "W": W,
                     "accuracy": m["accuracy"], "precision_macro": m["precision_macro"],
                     "recall_macro": m["recall_macro"], "f1_macro": m["f1_macro"],
                     "delta_acc": d, "delta_acc_pct": dp, "seed": cfg.experiment.seed})
        _confmat_fig(m["confusion"], names, f"{exp} | test {rname} | acc={m['accuracy']:.3f}",
                     run_dir / "figures" / fig_name(exp, "confmat", train=cfg.resolution.train, test=rname))
        print(f"  {rname:>9} ({H}x{W}): acc={m['accuracy']:.4f} f1={m['f1_macro']:.4f} ΔA={d:+.4f}")
    df = pd.DataFrame(rows)
    df.to_csv(run_dir / "metrics" / "eval_by_resolution.csv", index=False)
    fig, ax = plt.subplots(figsize=(5.5, 3.6))
    dfp = df.sort_values(["H", "W"])
    ax.plot(dfp["R_test"], dfp["accuracy"], "o-"); ax.set_ylim(0, 1.02); ax.grid(alpha=.3)
    ax.set_xlabel("Resolución de prueba"); ax.set_ylabel("Accuracy")
    ax.set_title(f"{exp} (train {cfg.resolution.train}, sin fine-tuning)", fontsize=9)
    fig.tight_layout(); fig.savefig(run_dir / "figures" / fig_name(exp, "acc_vs_resolution", train=cfg.resolution.train), dpi=150)
    plt.close(fig)
    return df
