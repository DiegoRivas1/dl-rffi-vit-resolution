"""Entrena (fase 1: ViT) y evalúa en todas las resoluciones.

    python -m scripts.train --config configs/experiments/f1_vit_base_r64.yaml
    python -m scripts.train --config ... --resume        # continuar tras una pausa
    python -m scripts.train --config ... --overwrite     # empezar de cero
"""
import argparse
import shutil
from pathlib import Path
import torch

from src.evaluation.eval_resolutions import run_eval
from src.models import build_model
from src.training.checkpoint import load_checkpoint
from src.training.data import build_dataset, make_loader, res_hw
from src.training.setup import load_all
from src.training.trainer import Trainer
from src.utils.config import load_config, dump_config
from src.utils.seed import set_seed
from src.evaluation.plots import plot_training_curves


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--no-eval", action="store_true")
    ap.add_argument("--device", default=None)
    a = ap.parse_args()

    cfg = load_config(a.config)
    device = torch.device(a.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    run_dir = Path(cfg.output.runs_dir) / cfg.experiment.name
    if a.overwrite and run_dir.exists():
        shutil.rmtree(run_dir)
    if not (a.resume and (run_dir / "config_snapshot.yaml").exists()):
        dump_config(cfg, run_dir / "config_snapshot.yaml")
    set_seed(cfg.experiment.seed)

    pre, subset, splits = load_all(cfg)
    tr_res = pre.resolutions[cfg.resolution.train]
    hw = res_hw(tr_res, pre.stft.signal_len)
    print(f"Dispositivo: {device} | R_train={cfg.resolution.train} {hw} | "
          f"train/val/test = {len(splits['train'])}/{len(splits['val'])}/{len(splits['test'])}")
    pin = device.type == "cuda"
    tr = make_loader(build_dataset(subset, splits["train"], tr_res, pre.stft, device), cfg.training.batch_size, True, cfg.training.num_workers, pin)
    va = make_loader(build_dataset(subset, splits["val"], tr_res, pre.stft, device), 256, False, cfg.training.num_workers, pin)
    model = build_model(cfg.model, len(subset["tx_names"]), hw)
    Trainer(model, cfg, run_dir, device, tr, va).fit(resume=a.resume)
    (run_dir / "figures").mkdir(parents=True, exist_ok=True)
    plot_training_curves(run_dir, cfg.experiment.name)

    if not a.no_eval:
        ck = load_checkpoint(run_dir / "checkpoints" / "best.pt", device)
        model.load_state_dict(ck["model"]); model.to(device)
        print(f"\nEvaluación sin fine-tuning (best.pt, época {ck['epoch']}):")
        run_eval(cfg, pre, subset, splits, model, device, run_dir)


if __name__ == "__main__":
    main()
