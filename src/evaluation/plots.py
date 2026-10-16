# src/evaluation/plots.py  (archivo nuevo)
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from src.utils.naming import fig_name


def plot_training_curves(run_dir, exp):
    run_dir = Path(run_dir)
    df = pd.read_csv(run_dir / "metrics" / "train_log.csv")
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9, 3.4))
    a1.plot(df.epoch, df.train_loss, label="train"); a1.plot(df.epoch, df.val_loss, label="val")
    a1.set_yscale("log"); a1.set_xlabel("Época"); a1.set_ylabel("Loss"); a1.legend(); a1.grid(alpha=.3)
    a2.plot(df.epoch, df.train_acc, label="train"); a2.plot(df.epoch, df.val_acc, label="val")
    a2.set_ylim(0, 1.02); a2.set_xlabel("Época"); a2.set_ylabel("Accuracy"); a2.legend(); a2.grid(alpha=.3)
    fig.suptitle(exp, fontsize=9); fig.tight_layout()
    fig.savefig(run_dir / "figures" / fig_name(exp, "training_curves", epochs=len(df)), dpi=150)
    plt.close(fig)