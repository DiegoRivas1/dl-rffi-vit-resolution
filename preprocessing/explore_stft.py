"""Exploración (aislada, sin entrenar) para fijar R_train y R_test.

    python -m preprocessing.explore_stft                       # tabla de configuraciones + demo sintética
    python -m preprocessing.explore_stft --config configs/preprocess/manysig_eq1.yaml   # + espectrogramas reales
    python -m preprocessing.explore_stft --pkl data/raw/ManySig.pkl                     # + chequeo equalized 0 vs 1

Muestra qué (F, T) salen de señales de 256 muestras, cuáles son compatibles con el
tamaño de parche, y cuánta resolución frecuencial REAL hay (1/win_len) frente a bins interpolados.
"""
import argparse
import itertools
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from preprocessing.stft_generator import compute_spectrograms, resolution_shape
from src.utils.config import load_config
from src.utils.naming import fig_name


def grid_table(signal_len=256, patch=8, n_ffts=(16, 32, 48, 64, 96, 128, 192, 256),
               wins=(16, 32, 48, 64, 96, 128, 192, 256), hops=(1, 2, 4, 8, 16, 32)):
    rows = []
    for nf, wl, hp in itertools.product(n_ffts, wins, hops):
        if wl > nf:
            continue
        F, T = resolution_shape(signal_len, nf, hp)
        if F % patch or T % patch:
            continue
        rows.append({"n_fft": nf, "win": wl, "hop": hp, "F": F, "T": T, "tokens": (F // patch) * (T // patch),
                     "zero_pad_x": nf / wl, "overlap_%": round(100 * (1 - hp / wl)) if hp < wl else 0})
    return rows


def print_table(rows, only_natural=True):
    print(f"{'n_fft':>6} {'win':>5} {'hop':>4} {'F':>4}x{'T':<4} {'tokens':>6} {'zeropad':>7} {'overlap%':>8}")
    for r in rows:
        if only_natural and r["zero_pad_x"] != 1:
            continue
        print(f"{r['n_fft']:>6} {r['win']:>5} {r['hop']:>4} {r['F']:>4}x{r['T']:<4} {r['tokens']:>6} {r['zero_pad_x']:>7.1f} {r['overlap_%']:>8}")


def synthetic_iq(n=3, N=256, seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(N)
    out = []
    for k in range(n):
        z = np.exp(1j * 2 * np.pi * (0.05 + 0.08 * k) * t) + 0.2 * (rng.normal(size=N) + 1j * rng.normal(size=N))
        out.append(np.stack([z.real, z.imag], -1))
    return np.stack(out).astype(np.float32)


def plot_grid(iq, labels, res_cfg, stft_cfg, out_path):
    names = list(res_cfg.keys())
    fig, axes = plt.subplots(len(iq), len(names), figsize=(3 * len(names), 2.6 * len(iq)), squeeze=False)
    for j, n in enumerate(names):
        r = res_cfg[n]
        s = compute_spectrograms(torch.from_numpy(iq), r["n_fft"], r["win_len"], r["hop"], window=stft_cfg.window,
                                 log_scale=stft_cfg.log_scale, normalize=True).numpy()
        for i in range(len(iq)):
            ax = axes[i][j]
            ax.imshow(s[i], aspect="auto", origin="lower", cmap="viridis")
            ax.set_xticks([]); ax.set_yticks([])
            if i == 0:
                ax.set_title(f"{n}\n{s.shape[1]}x{s.shape[2]}", fontsize=8)
            if j == 0:
                ax.set_ylabel(labels[i], fontsize=8)
    fig.tight_layout(); fig.savefig(out_path, dpi=130); plt.close(fig)
    print("figura:", out_path)


def eq_check(pkl):
    from preprocessing.dataset_loader import load_manysig
    d = load_manysig(pkl)
    print("\nChequeo equalized 0 vs 1 (Tx0, Rx0, fecha0, primeras 200 señales):")
    for eq in d["equalized_list"]:
        a = np.asarray(d["data"][0][0][0][d["equalized_list"].index(eq)])[:200]
        p = (a ** 2).sum(-1)
        print(f"  equalized={eq}: shape={a.shape} potencia_media={p.mean():.4f} std={a.std():.4f} "
              f"max|x|={np.abs(a).max():.3f} NaN={int(np.isnan(a).sum())} señales_planas={(p.std(1) < 1e-9).sum()}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--signal-len", type=int, default=256)
    ap.add_argument("--patch", type=int, default=8)
    ap.add_argument("--all", action="store_true", help="incluir también configuraciones con zero-padding (n_fft > win)")
    ap.add_argument("--config", default=None, help="YAML de preprocesamiento (para graficar sus resoluciones)")
    ap.add_argument("--pkl", default=None)
    ap.add_argument("--out", default="figures/exploration")
    a = ap.parse_args()

    rows = grid_table(a.signal_len, a.patch)
    print(f"Configuraciones compatibles con parche {a.patch} y señales de {a.signal_len} muestras "
          f"({'todas' if a.all else 'solo n_fft = win, sin zero-padding'}):")
    print_table(rows, only_natural=not a.all)

    cfg = load_config(a.config or "configs/preprocess/manysig_eq1.yaml")
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    sub_path = Path("data/processed") / f"{cfg.name}_subset.npz"
    if sub_path.exists():
        from preprocessing.dataset_loader import load_subset
        s = load_subset(sub_path)
        idx = [int(np.where(s["y"] == c)[0][0]) for c in range(min(3, len(s["tx_names"])))]
        iq, labels, tag = s["X"][idx], [f"Tx {s['tx_names'][c]}" for c in range(len(idx))], "real"
    else:
        iq, labels, tag = synthetic_iq(), ["sint. 0", "sint. 1", "sint. 2"], "synthetic"
        print("\n(no hay subset procesado: se usan señales sintéticas; corre scripts.preprocess para ver las reales)")
    plot_grid(iq, labels, cfg.resolutions, cfg.stft, out / fig_name("explore", "spectrogram_grid", data=tag))
    if a.pkl:
        eq_check(a.pkl)


if __name__ == "__main__":
    main()
