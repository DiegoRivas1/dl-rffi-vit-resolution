"""Lectura de ManySig.pkl (WiSig) y extracción de un subconjunto controlado.

Estructura oficial:  data["data"][tx][rx][fecha][equalized] -> ndarray (1000, 256, 2)
(columna 0 = I, columna 1 = Q; entrada del clasificador oficial: Input(shape=(256,2))).
"""
import pickle
from pathlib import Path
import numpy as np


def load_manysig(pkl_path):
    with open(pkl_path, "rb") as f:
        return pickle.load(f)


def build_subset(data, equalized=1, tx_list=None, rx_list=None, date_list=None, n_per_cell=1000):
    tx_all, rx_all = data["tx_list"], data["rx_list"]
    dt_all, eq_all = data["capture_date_list"], data["equalized_list"]
    tx_sel = tx_list or tx_all
    rx_sel = rx_list or rx_all
    dt_sel = date_list or dt_all
    eq_i = eq_all.index(equalized)

    X, y, rx_id, dt_id = [], [], [], []
    for cls, tx in enumerate(tx_sel):
        ti = tx_all.index(tx)
        for rx in rx_sel:
            ri = rx_all.index(rx)
            for dt in dt_sel:
                di = dt_all.index(dt)
                arr = np.asarray(data["data"][ti][ri][di][eq_i])
                if arr.ndim != 3 or arr.shape[0] == 0:
                    continue  # celda ausente
                arr = arr[:n_per_cell].astype(np.float32)
                X.append(arr)
                y.append(np.full(len(arr), cls, dtype=np.int64))
                rx_id.append(np.full(len(arr), ri, dtype=np.int16))
                dt_id.append(np.full(len(arr), di, dtype=np.int16))
    if not X:
        raise ValueError("El subconjunto quedó vacío: revisar tx/rx/fecha/equalized.")
    y = np.concatenate(y)
    for c in range(len(tx_sel)):
        if not (y == c).any():
            raise ValueError(f"El transmisor {tx_sel[c]} no tiene señales en la selección.")
    return {
        "X": np.concatenate(X), "y": y,
        "rx_id": np.concatenate(rx_id), "date_id": np.concatenate(dt_id),
        "tx_names": np.array(tx_sel), "rx_names": np.array(rx_all), "date_names": np.array(dt_all),
        "equalized": np.array(equalized),
    }


def save_subset(subset, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **subset)


def load_subset(path):
    z = np.load(path, allow_pickle=False)
    return {k: z[k] for k in z.files}


if __name__ == "__main__":
    # Prueba aislada con datos sintéticos mínimos.
    fake = {"tx_list": ["a", "b"], "rx_list": ["r1", "r2"], "capture_date_list": ["d1"],
            "equalized_list": [0, 1],
            "data": [[[[np.random.randn(10, 256, 2), np.random.randn(10, 256, 2)]] for _ in range(2)] for _ in range(2)]}
    s = build_subset(fake, equalized=1, rx_list=["r1"], n_per_cell=5)
    assert s["X"].shape == (10, 256, 2) and s["X"].dtype == np.float32 and set(s["y"]) == {0, 1}
    print("dataset_loader OK", s["X"].shape)
