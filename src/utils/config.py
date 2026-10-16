"""Carga de YAML con acceso por atributos (cfg.training.lr)."""
from pathlib import Path
import yaml


class Cfg(dict):
    def __getattr__(self, k):
        try:
            return self[k]
        except KeyError as e:
            raise AttributeError(k) from e

    def __setattr__(self, k, v):
        self[k] = v


def _wrap(o):
    if isinstance(o, dict):
        return Cfg({k: _wrap(v) for k, v in o.items()})
    if isinstance(o, list):
        return [_wrap(v) for v in o]
    return o


def _plain(o):
    if isinstance(o, dict):
        return {k: _plain(v) for k, v in o.items()}
    if isinstance(o, list):
        return [_plain(v) for v in o]
    return o


def load_config(path):
    with open(path, "r", encoding="utf-8") as f:
        return _wrap(yaml.safe_load(f))


def dump_config(cfg, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(_plain(cfg), f, sort_keys=False, allow_unicode=True)
