"""Re-evalúa un modelo entrenado en las resoluciones del YAML.

    python -m scripts.evaluate --config configs/experiments/f1_vit_base_r64.yaml [--ckpt best|last]
"""
import argparse
from pathlib import Path
import torch
from src.evaluation.eval_resolutions import run_eval
from src.models import build_model
from src.training.checkpoint import load_checkpoint
from src.training.data import res_hw
from src.training.setup import load_all
from src.utils.config import load_config


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt", default="best", choices=["best", "last"])
    ap.add_argument("--device", default=None)
    a = ap.parse_args()
    cfg = load_config(a.config)
    device = torch.device(a.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    run_dir = Path(cfg.output.runs_dir) / cfg.experiment.name
    pre, subset, splits = load_all(cfg)
    hw = res_hw(pre.resolutions[cfg.resolution.train], pre.stft.signal_len)
    model = build_model(cfg.model, len(subset["tx_names"]), hw)
    model.load_state_dict(load_checkpoint(run_dir / "checkpoints" / f"{a.ckpt}.pt", device)["model"])
    model.to(device)
    run_eval(cfg, pre, subset, splits, model, device, run_dir)


if __name__ == "__main__":
    main()
