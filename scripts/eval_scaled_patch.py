"""Parche FIJO vs parche ESCALADO con la resolución, para ViT y MSViT (sin reentrenar, mismos best.pt).

Parche fijo    : comportamiento original. La rejilla de tokens cambia con la resolución (el ViT interpola su pos_embed).
Parche escalado: el parche crece con la resolución y la rejilla de tokens es siempre la de entrenamiento.
                 ViT: kernel reescalado en el dominio DCT (src/models/patch_scale.py). MSViT: SCConv.scaled_basis.

Cada modelo se evalúa con tres tipos de entrada:
  STFT regenerada (ventana = n_fft) : las resoluciones de la tesis; cambia el compromiso tiempo-frecuencia.
  STFT de ventana fija              : `resolutions_fixed_window` del YAML; misma ventana que en entrenamiento, otro muestreo.
  redimensionada                    : el espectrograma de R_train reescalado por interpolación.

    python -m scripts.eval_scaled_patch --runs f1_vit_base_r64 f1_vit_base_r64_s43 ... f3_msvit_base_r64 f3_msvit_base_r64_s43 ...
"""
import argparse
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import TensorDataset

from scripts.eval_resize import resize_batch
from src.evaluation.eval_resolutions import predict
from src.evaluation.metrics import compute_metrics
from src.models import build_model
from src.training.checkpoint import load_checkpoint
from src.training.data import build_dataset, make_loader, res_hw
from src.training.setup import load_all
from src.utils.config import load_config

REGEN, FIXED = "STFT regenerada (ventana = n_fft)", "STFT de ventana fija"


def eval_run(run, runs_dir, mode, device):
    run_dir = Path(runs_dir) / run
    cfg = load_config(run_dir / "config_snapshot.yaml")
    pre, subset, splits = load_all(cfg)
    n_cls = len(subset["tx_names"])
    r_tr = cfg.resolution.train
    hw_tr = tuple(res_hw(pre.resolutions[r_tr], pre.stft.signal_len))
    model = build_model(cfg.model, n_cls, hw_tr)
    model.load_state_dict(load_checkpoint(run_dir / "checkpoints" / "best.pt", device)["model"])
    model.to(device).eval()

    def stft(res):
        return build_dataset(subset, splits["test"], res, pre.stft, device).tensors

    X_tr, Y = stft(pre.resolutions[r_tr])
    cases = [(r_tr, hw_tr, REGEN, X_tr)]                                   # (etiqueta, forma, entrada, tensor)
    for rname in [r for r in cfg.resolution.test if r != r_tr]:
        hw = tuple(res_hw(pre.resolutions[rname], pre.stft.signal_len))
        cases.append((rname, hw, REGEN, stft(pre.resolutions[rname])[0]))
        cases.append((rname, hw, f"redimensionada ({mode})", resize_batch(X_tr, hw, mode, True)))
    for rname, res in (pre.get("resolutions_fixed_window") or {}).items():
        hw = tuple(res_hw(res, pre.stft.signal_len))
        cases.append((rname.split("_", 1)[-1], hw, FIXED, stft(res)[0]))

    rows = []
    for label, hw, in_name, X in cases:
        for patch, ref in (("fijo", None), ("escalado", hw_tr)):
            model.ref_hw = ref
            yt, yp = predict(model, make_loader(TensorDataset(X, Y), 256, False), device)
            m = compute_metrics(yt, yp, n_cls)
            rows.append({"experiment": run, "model": cfg.model.type, "seed": cfg.experiment.seed, "R_train": r_tr,
                         "R_test": label, "H": hw[0], "W": hw[1], "entrada": in_name, "parche": patch,
                         "accuracy": m["accuracy"], "f1_macro": m["f1_macro"]})
            print(f"  {run:>24} | {label:>9} | {in_name:<34} | parche {patch:<8}: acc={m['accuracy']:.4f}")
    model.ref_hw = None
    df = pd.DataFrame(rows)
    (run_dir / "metrics").mkdir(parents=True, exist_ok=True)
    df.to_csv(run_dir / "metrics" / f"eval_scaled_patch_{mode}.csv", index=False)
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--runs-dir", default="runs")
    ap.add_argument("--mode", default="bilinear", choices=["bilinear", "bicubic", "nearest"])
    ap.add_argument("--device", default=None)
    a = ap.parse_args()
    device = torch.device(a.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    df = pd.concat([eval_run(r, a.runs_dir, a.mode, device) for r in a.runs], ignore_index=True)
    g = df.groupby(["entrada", "H", "W", "R_test", "model", "parche"])["accuracy"]
    summ = g.agg(["mean", "std", "count"]).round(4).reset_index().drop(columns=["H", "W"])
    out = Path("figures/comparison"); out.mkdir(parents=True, exist_ok=True)
    path = out / f"scaled_patch__summary__mode-{a.mode}_n_runs-{df['experiment'].nunique()}.csv"
    summ.to_csv(path, index=False)
    wide = summ.assign(v=summ["mean"].map("{:.3f}".format) + " ± " + summ["std"].fillna(0).map("{:.3f}".format))
    wide = wide.pivot_table(index=["entrada", "R_test"], columns=["model", "parche"], values="v", aggfunc="first", sort=False)
    print("\nRESUMEN (media ± desv. entre runs):\n", wide.to_string(), "\n\ntabla:", path)


if __name__ == "__main__":
    main()
