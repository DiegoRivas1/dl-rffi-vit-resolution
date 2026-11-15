"""¿Cuánto del ~100 % depende del receptor y del día de entrenamiento? (sin reentrenar, en R_train)

Toma modelos entrenados con UN receptor y UNA fecha y los evalúa, en la misma resolución de entrenamiento, con:
  - la celda de entrenamiento (solo sus señales de prueba),
  - el mismo receptor en las otras fechas,
  - los otros receptores en la misma fecha.
Necesita el pkl original (lee celdas que no están en el subconjunto procesado).

    python -m scripts.eval_cross_condition --runs f1_vit_base_r64 f1_vit_base_r64_s43 ... f3_msvit_base_r64 ...
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from preprocessing.dataset_loader import build_subset, load_manysig
from src.evaluation.eval_resolutions import predict
from src.evaluation.metrics import compute_metrics
from src.models import build_model
from src.training.checkpoint import load_checkpoint
from src.training.data import build_dataset, make_loader, res_hw
from src.training.setup import load_all
from src.utils.config import load_config

SAME, DATE, RX = "celda de entrenamiento (prueba)", "mismo receptor, otra fecha", "otro receptor, misma fecha"


def eval_run(run, runs_dir, data, n, device):
    run_dir = Path(runs_dir) / run
    cfg = load_config(run_dir / "config_snapshot.yaml")
    pre, subset, splits = load_all(cfg)
    d = pre.dataset
    if not d.rx_list or not d.date_list or len(d.rx_list) != 1 or len(d.date_list) != 1:
        raise SystemExit(f"{run}: este control es para modelos entrenados con un solo receptor y una sola fecha.")
    rx0, dt0 = d.rx_list[0], d.date_list[0]
    tx = [str(t) for t in subset["tx_names"]]
    res = pre.resolutions[cfg.resolution.train]
    model = build_model(cfg.model, len(tx), tuple(res_hw(res, pre.stft.signal_len)))
    model.load_state_dict(load_checkpoint(run_dir / "checkpoints" / "best.pt", device)["model"])
    model.to(device).eval()

    def acc(sub, idx):
        yt, yp = predict(model, make_loader(build_dataset(sub, idx, res, pre.stft, device), 256, False), device)
        return compute_metrics(yt, yp, len(tx))["accuracy"]

    cells = [(SAME, rx0, dt0, acc(subset, splits["test"]))]
    others = [(DATE, rx0, dt) for dt in data["capture_date_list"] if dt != dt0] + \
             [(RX, rx, dt0) for rx in data["rx_list"] if rx != rx0]
    for cond, rx, dt in others:
        try:
            sub = build_subset(data, d.equalized, tx, [rx], [dt], n)
        except ValueError:                       # celda sin señales de algún transmisor
            print(f"  {run:>24} | {cond:<28} | rx {rx:>5} | {dt}: sin datos, se omite"); continue
        cells.append((cond, rx, dt, acc(sub, np.arange(len(sub["y"])))))
    rows = [{"experiment": run, "model": cfg.model.type, "seed": cfg.experiment.seed, "condicion": c,
             "rx": rx, "fecha": dt, "accuracy": a} for c, rx, dt, a in cells]
    for r in rows:
        print(f"  {run:>24} | {r['condicion']:<31} | rx {r['rx']:>5} | {r['fecha']}: acc={r['accuracy']:.4f}")
    df = pd.DataFrame(rows)
    (run_dir / "metrics").mkdir(parents=True, exist_ok=True)
    df.to_csv(run_dir / "metrics" / "eval_cross_condition.csv", index=False)
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--runs-dir", default="runs")
    ap.add_argument("--pkl", default="data/raw/ManySig.pkl")
    ap.add_argument("--n", type=int, default=250, help="señales por transmisor en cada celda (receptor, fecha)")
    ap.add_argument("--device", default=None)
    a = ap.parse_args()
    device = torch.device(a.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Cargando {a.pkl} ...")
    data = load_manysig(a.pkl)
    df = pd.concat([eval_run(r, a.runs_dir, data, a.n, device) for r in a.runs], ignore_index=True)
    out = Path("figures/comparison"); out.mkdir(parents=True, exist_ok=True)
    n = df["experiment"].nunique()
    per_run = df.groupby(["model", "experiment", "condicion"])["accuracy"].mean().reset_index()   # media sobre celdas
    summ = per_run.groupby(["condicion", "model"])["accuracy"].agg(["mean", "std", "count"]).round(4).reset_index()
    cells = df.groupby(["condicion", "rx", "fecha", "model"])["accuracy"].agg(["mean", "std"]).round(4).reset_index()
    summ.to_csv(out / f"cross_condition__summary__n_runs-{n}.csv", index=False)
    cells.to_csv(out / f"cross_condition__by_cell__n_runs-{n}.csv", index=False)
    print("\nRESUMEN (media ± desv. entre runs; cada run promedia sus celdas):\n", summ.to_string(index=False))
    print("\nPOR CELDA (media entre runs):\n",
          cells.pivot_table(index=["condicion", "rx", "fecha"], columns="model", values="mean", sort=False).to_string())
    print("\ntablas:", out / f"cross_condition__summary__n_runs-{n}.csv", "y", out / f"cross_condition__by_cell__n_runs-{n}.csv")


if __name__ == "__main__":
    main()
