"""ManySig.pkl -> subconjunto I/Q (float32) + particiones. Se ejecuta una vez por configuración.

    python -m scripts.preprocess --config configs/preprocess/manysig_eq1.yaml [--pkl RUTA]
"""
import argparse
import numpy as np
from preprocessing.dataset_loader import load_manysig, build_subset, save_subset
from preprocessing.splits import stratified_split, save_splits
from src.training.setup import processed_paths
from src.utils.config import load_config


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--pkl", default=None, help="sobrescribe dataset.pkl_path")
    a = ap.parse_args()
    pre = load_config(a.config)
    pkl = a.pkl or pre.dataset.pkl_path
    print(f"Cargando {pkl} ...")
    data = load_manysig(pkl)
    d = pre.dataset
    sub = build_subset(data, d.equalized, d.tx_list, d.rx_list, d.date_list, d.n_per_cell)
    assert sub["X"].shape[1:] == (pre.stft.signal_len, 2), sub["X"].shape
    spl = stratified_split(sub["y"], pre.split.train, pre.split.val, pre.split.test, pre.split.seed)
    p_sub, p_spl = processed_paths(pre)
    save_subset(sub, p_sub); save_splits(spl, p_spl)
    print(f"Subconjunto: X{sub['X'].shape}, clases={[str(t) for t in sub['tx_names']]}")
    print("Señales por clase:", np.bincount(sub["y"]).tolist())
    print({k: len(v) for k, v in spl.items()})
    print(f"Guardado: {p_sub}\n          {p_spl}")


if __name__ == "__main__":
    main()
