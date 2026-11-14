"""MSViT: parche FIJO vs parche ESCALADO con la resolución (sin reentrenar, mismos best.pt).

Parche fijo    : el stem usa siempre 4x4 -> la rejilla de tokens crece/decrece con la resolución (lo evaluado hasta ahora).
Parche escalado: el parche crece con la resolución (R32x32 -> 2x2, R96x64 -> 6x4, R128x128 -> 8x8) y se proyecta a los
                 mismos 16 canales DCT -> la rejilla de tokens es siempre la de entrenamiento (16x16). No hay pesos nuevos.

Se evalúa con STFT regenerada y con redimensionamiento de R_train, para cada run:

    python -m scripts.eval_msvit_scale --runs f3_msvit_base_r64 f3_msvit_base_r64_s43 f3_msvit_base_r64_s44 f3_msvit_base_r64_s45 f3_msvit_base_r64_s46
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


def eval_run(run, runs_dir, mode, device):
    run_dir = Path(runs_dir) / run
    cfg = load_config(run_dir / "config_snapshot.yaml")
    if cfg.model.type != "msvit":
        raise SystemExit(f"{run}: este script es solo para runs de MSViT (model.type = {cfg.model.type}).")
    pre, subset, splits = load_all(cfg)
    n_cls = len(subset["tx_names"])
    r_tr = cfg.resolution.train
    hw_tr = tuple(res_hw(pre.resolutions[r_tr], pre.stft.signal_len))
    model = build_model(cfg.model, n_cls, hw_tr)
    model.load_state_dict(load_checkpoint(run_dir / "checkpoints" / "best.pt", device)["model"])
    model.to(device).eval()
    X_tr, Y = build_dataset(subset, splits["test"], pre.resolutions[r_tr], pre.stft, device).tensors
    rows = []
    for rname in [r_tr] + [r for r in cfg.resolution.test if r != r_tr]:
        hw = tuple(res_hw(pre.resolutions[rname], pre.stft.signal_len))
        inputs = {"STFT regenerada": build_dataset(subset, splits["test"], pre.resolutions[rname], pre.stft, device).tensors[0]}
        if hw != hw_tr:
            inputs[f"redimensionada ({mode})"] = resize_batch(X_tr, hw, mode, True)
        for in_name, X in inputs.items():
            for stem, ref in (("fijo", None), ("escalado", hw_tr)):
                model.ref_hw = ref
                yt, yp = predict(model, make_loader(TensorDataset(X, Y), 256, False), device)
                m = compute_metrics(yt, yp, n_cls)
                rows.append({"experiment": run, "seed": cfg.experiment.seed, "R_train": r_tr, "R_test": rname,
                             "H": hw[0], "W": hw[1], "entrada": in_name, "parche": stem,
                             "accuracy": m["accuracy"], "f1_macro": m["f1_macro"]})
                print(f"  {run:>24} | {rname:>9} | {in_name:<26} | parche {stem:<8}: acc={m['accuracy']:.4f}")
    model.ref_hw = None
    df = pd.DataFrame(rows)
    (run_dir / "metrics").mkdir(parents=True, exist_ok=True)
    df.to_csv(run_dir / "metrics" / f"eval_scale_stem_{mode}.csv", index=False)
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
    summ = (df.groupby(["entrada", "R_test", "H", "W", "parche"])["accuracy"].agg(["mean", "std", "count"]).round(4)
              .reset_index().sort_values(["entrada", "H", "W", "parche"]).drop(columns=["H", "W"]))
    out = Path("figures/comparison"); out.mkdir(parents=True, exist_ok=True)
    path = out / f"msvit_scale_stem__summary__mode-{a.mode}_n-{df['experiment'].nunique()}.csv"
    summ.to_csv(path, index=False)
    print("\nRESUMEN (media ± desv. entre runs):\n", summ.to_string(index=False), "\ntabla:", path)


if __name__ == "__main__":
    main()
