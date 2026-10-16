"""Genera un ManySig.pkl SINTÉTICO pequeño (misma estructura) para probar el pipeline sin el archivo de 2.3 GB.

    python -m scripts.make_synthetic_pkl --out data/raw/ManySig_synth.pkl
Cada Tx tiene un desfase de frecuencia y desbalance I/Q distintos -> clases separables.
"""
import argparse
import pickle
from pathlib import Path
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/raw/ManySig_synth.pkl")
    ap.add_argument("--n", type=int, default=200)
    a = ap.parse_args()
    rng = np.random.default_rng(0)
    tx = ["14-10", "14-7", "20-15", "20-19", "6-15", "8-20"]
    rx = ["1-1", "1-19"]; dates = ["2021_03_01", "2021_03_08"]
    n = np.arange(256)
    data = []
    for t in range(len(tx)):
        f0, g = 0.01 + 0.02 * t, 1.0 + 0.05 * t
        per_rx = []
        for _ in rx:
            per_date = []
            for _ in dates:
                cells = []
                for _eq in (0, 1):
                    sig = np.empty((a.n, 256, 2))
                    for i in range(a.n):
                        ph = rng.uniform(0, 2 * np.pi)
                        z = np.exp(1j * (2 * np.pi * (0.1 + f0 * rng.normal(1, .02)) * n + ph))
                        z = z + 0.1 * rng.normal(size=256) + 0.1j * rng.normal(size=256)
                        sig[i, :, 0], sig[i, :, 1] = z.real, g * z.imag
                    cells.append(sig)
                per_date.append(cells)
            per_rx.append(per_date)
        data.append(per_rx)
    out = {"tx_list": tx, "rx_list": rx, "capture_date_list": dates, "equalized_list": [0, 1],
           "max_sig": a.n, "data": data}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "wb") as f:
        pickle.dump(out, f)
    print("escrito", a.out)


if __name__ == "__main__":
    main()
