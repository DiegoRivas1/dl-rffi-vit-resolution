import pickle
from pathlib import Path

DATASET = Path("data/raw/ManySig.pkl")

print("=" * 70)
print("INSPECCIÓN DETALLADA DE ManySig.pkl")
print("=" * 70)

with open(DATASET, "rb") as f:
    data = pickle.load(f)

# ----------------------------------------------------------
# METADATOS
# ----------------------------------------------------------

print("\n[1] METADATOS")
print("-" * 70)

for key in [
    "tx_list",
    "rx_list",
    "capture_date_list",
    "equalized_list",
    "max_sig"
]:
    print(f"{key}: {data[key]}")

# ----------------------------------------------------------
# ESTRUCTURA DE data
# ----------------------------------------------------------

signals = data["data"]

print("\n[2] ESTRUCTURA DE data")
print("-" * 70)

print("type(data['data']):", type(signals))
print("len(data['data']):", len(signals))

obj = signals

for level in range(6):

    print(f"\nNivel {level}:")
    print("  tipo:", type(obj))

    try:
        print("  longitud:", len(obj))
    except TypeError:
        pass

    if hasattr(obj, "shape"):
        print("  shape:", obj.shape)

    if hasattr(obj, "dtype"):
        print("  dtype:", obj.dtype)

    # Si llegamos a ndarray, mostramos una muestra pequeña
    if hasattr(obj, "shape"):
        print("  primeros valores:", obj.reshape(-1)[:5])
        break

    # Descender únicamente por el primer elemento
    if isinstance(obj, (list, tuple)) and len(obj) > 0:
        obj = obj[0]
    else:
        break

print("\n" + "=" * 70)
print("FIN")
print("=" * 70)