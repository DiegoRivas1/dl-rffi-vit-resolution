import os
import time
from pathlib import Path
import torch


def save_checkpoint(state, path, retries=8, wait=0.4):
    """Escritura atómica con reintentos (Windows: antivirus/indexador pueden bloquear el .tmp)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(state, tmp)
    for i in range(retries):
        try:
            os.replace(tmp, path)
            return True
        except PermissionError:
            time.sleep(wait * (i + 1))
    try:  # último recurso: escritura directa (no atómica)
        torch.save(state, path)
        if tmp.exists():
            tmp.unlink()
        return True
    except OSError as e:
        print(f"[warning] no se pudo guardar {path.name}: {e}. Se intentará de nuevo en el próximo guardado.")
        return False


def load_checkpoint(path, map_location="cpu"):
    return torch.load(path, map_location=map_location, weights_only=False)