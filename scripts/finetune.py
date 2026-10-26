"""FASE 2: fine-tuning del ViT de la fase 1 en una resolución objetivo.

    python -m scripts.finetune --config configs/experiments/f2_vit_ft_r32x32.yaml
    python -m scripts.finetune --config ... --resume | --overwrite

Parte de runs/<source_run>/checkpoints/<source_ckpt>.pt, adapta la cuadrícula del positional embedding
a la resolución objetivo y reentrena con las señales de ENTRENAMIENTO (mismas particiones que la fase 1).
"""
import argparse
import shutil
from pathlib import Path
import torch

from src.evaluation.eval_finetune import run_eval_ft
from src.models import build_model
from src.training.checkpoint import load_checkpoint
from src.training.data import build_dataset, make_loader, res_hw
from src.training.setup import load_all
from src.training.trainer import Trainer
from src.utils.config import load_config, dump_config
from src.utils.seed import set_seed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--overwrite", action="store_true")
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
    src_dir = Path(cfg.output.runs_dir) / cfg.finetune.source_run
    src_cfg = load_config(src_dir / "config_snapshot.yaml")
    assert src_cfg.resolution.train == cfg.resolution.train, "R_train del YAML != R_train del modelo fuente"
    n_cls = len(subset["tx_names"])
    hw_src = res_hw(pre.resolutions[cfg.resolution.train], pre.stft.signal_len)
    hw_ft = res_hw(pre.resolutions[cfg.resolution.finetune], pre.stft.signal_len)

    model = build_model(src_cfg.model, n_cls, hw_src)             # misma arquitectura que la fuente
    ck = load_checkpoint(src_dir / "checkpoints" / f"{cfg.finetune.source_ckpt}.pt", device)
    model.load_state_dict(ck["model"])
    model.adapt_to(hw_ft)
    print(f"Fine-tuning: {cfg.finetune.source_run} ({cfg.resolution.train}, época {ck['epoch']}) -> "
          f"{cfg.resolution.finetune} {hw_ft} | dispositivo: {device}")

    ft_res = pre.resolutions[cfg.resolution.finetune]
    pin = device.type == "cuda"
    tr = make_loader(build_dataset(subset, splits["train"], ft_res, pre.stft, device), cfg.training.batch_size, True, cfg.training.num_workers, pin)
    va = make_loader(build_dataset(subset, splits["val"], ft_res, pre.stft, device), 256, False, cfg.training.num_workers, pin)
    Trainer(model, cfg, run_dir, device, tr, va).fit(resume=a.resume)

    try:
        from src.evaluation.plots import plot_training_curves
        (run_dir / "figures").mkdir(parents=True, exist_ok=True)
        plot_training_curves(run_dir, cfg.experiment.name)
    except ImportError:
        pass

    best = load_checkpoint(run_dir / "checkpoints" / "best.pt", device)
    model.load_state_dict(best["model"]); model.to(device)
    print(f"\nEvaluación CON fine-tuning en {cfg.resolution.finetune} (best.pt, época {best['epoch']}):")
    run_eval_ft(cfg, pre, subset, splits, model, device, run_dir)


if __name__ == "__main__":
    main()
