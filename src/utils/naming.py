"""Nombres de archivos: <experimento>__<tipo>__<caracteristicas>.png"""
import re


def _clean(s):
    return re.sub(r"[^A-Za-z0-9._\-]+", "-", str(s)).strip("-")


def fig_name(exp, kind, **chars):
    parts = [_clean(exp), _clean(kind)]
    if chars:
        parts.append("_".join(f"{_clean(k)}-{_clean(v)}" for k, v in chars.items()))
    return "__".join(parts) + ".png"


if __name__ == "__main__":
    print(fig_name("f1_vit_base_r64", "confmat", test="R96x64", acc="0.912"))
