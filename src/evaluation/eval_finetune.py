"""Evaluación del modelo con fine-tuning (fase 2). Se reporta APARTE: no es generalización zero-shot."""
from pathlib import Path
import pandas as pd

from src.evaluation.eval_resolutions import predict, _confmat_fig
from src.evaluation.metrics import compute_metrics, delta
from src.training.data import build_dataset, make_loader, res_hw
from src.utils.naming import fig_name


def load_source_reference(runs_dir, source_run, r_train):
    """Exactitud del modelo fuente (sin FT) en cada R_test y en R_train, leída de su CSV de la fase 1."""
    df = pd.read_csv(Path(runs_dir) / source_run / "metrics" / "eval_by_resolution.csv")
    df = df[df["R_train"] == r_train]
    zs = dict(zip(df["R_test"], df["accuracy"]))
    return zs[r_train], zs


def run_eval_ft(cfg, pre, subset, splits, model, device, run_dir):
    run_dir = Path(run_dir)
    (run_dir / "metrics").mkdir(parents=True, exist_ok=True)
    (run_dir / "figures").mkdir(parents=True, exist_ok=True)
    exp, names = cfg.experiment.name, [str(t) for t in subset["tx_names"]]
    r_src, r_ft = cfg.resolution.train, cfg.resolution.finetune
    ref_acc, zs = load_source_reference(cfg.output.runs_dir, cfg.finetune.source_run, r_src)
    order = [r_ft] + [r for r in cfg.resolution.test if r != r_ft]
    rows = []
    for rname in order:
        res = pre.resolutions[rname]
        ds = build_dataset(subset, splits["test"], res, pre.stft, device)
        yt, yp = predict(model, make_loader(ds, 256, False), device)
        m = compute_metrics(yt, yp, len(names))
        H, W = res_hw(res, pre.stft.signal_len)
        d, dp = delta(ref_acc, m["accuracy"])
        rows.append({"experiment": exp, "phase": cfg.experiment.phase, "model": "vit",
                     "finetune": True, "R_train": r_src, "R_finetune": r_ft, "R_test": rname, "H": H, "W": W,
                     "accuracy": m["accuracy"], "precision_macro": m["precision_macro"],
                     "recall_macro": m["recall_macro"], "f1_macro": m["f1_macro"],
                     "delta_acc": d, "delta_acc_pct": dp, "acc_zero_shot_source": zs.get(rname, float("nan")),
                     "seed": cfg.experiment.seed})
        tag = "TARGET" if rname == r_ft else "otra"
        print(f"  {rname:>9} ({H}x{W}) [{tag}]: acc={m['accuracy']:.4f} f1={m['f1_macro']:.4f} "
              f"(sin FT: {zs.get(rname, float('nan')):.4f})")
        _confmat_fig(m["confusion"], names, f"{exp} | FT {r_ft} | test {rname} | acc={m['accuracy']:.3f}",
                     run_dir / "figures" / fig_name(exp, "confmat", ft=r_ft, test=rname))
    df = pd.DataFrame(rows)
    df.to_csv(run_dir / "metrics" / "eval_by_resolution.csv", index=False)
    return df
