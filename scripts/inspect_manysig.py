"""Inspección de ManySig.pkl (tu main.py original, parametrizado)."""
import argparse
import pickle


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pkl", default="data/raw/ManySig.pkl")
    a = ap.parse_args()
    with open(a.pkl, "rb") as f:
        d = pickle.load(f)
    for k in ("tx_list", "rx_list", "capture_date_list", "equalized_list", "max_sig"):
        print(f"{k}: {d[k]}")
    x = d["data"]
    for lvl in range(4):
        print(f"nivel {lvl}: len={len(x)}"); x = x[0]
    print("hoja:", type(x), x.shape, x.dtype)


if __name__ == "__main__":
    main()
