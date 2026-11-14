"""Reescalado de un kernel de patch embedding a otro tamaño de parche, en el dominio DCT (sin pesos nuevos).

Un kernel K de (p x p) calcula el token como <K, parche>. Como la DCT 2D ortonormal conserva el producto interno,
<K, parche> = <DCT(K), DCT(parche)>. Si la imagen se reescala y el parche pasa a (ph x pw), los coeficientes DCT de
baja frecuencia del parche nuevo son (casi) los del original salvo el factor sqrt(ph/p)*sqrt(pw/p). Por eso el kernel
equivalente es: DCT(K) recortado o rellenado con ceros a (ph x pw), transformado de vuelta y multiplicado por
sqrt(p/ph)*sqrt(p/pw). Es la misma operación que SCConv.scaled_basis de MSViT, aplicada a un kernel aprendido.
"""
import math
import torch


def dct_matrix(n):
    """DCT-II ortonormal (n x n): M @ M.T = I."""
    k = torch.arange(n, dtype=torch.float32)[:, None]
    i = torch.arange(n, dtype=torch.float32)[None, :]
    M = torch.cos(math.pi * (2 * i + 1) * k / (2 * n)) * math.sqrt(2.0 / n)
    M[0] = M[0] / math.sqrt(2.0)
    return M


@torch.no_grad()
def rescale_patch_kernel(weight, patch):
    """weight: (D, C, p, p) -> (D, C, ph, pw)."""
    p, (ph, pw) = weight.shape[-1], patch
    M, Mh, Mw = (dct_matrix(n).to(weight.device, weight.dtype) for n in (p, ph, pw))
    coef = torch.einsum("ux,dcxy,vy->dcuv", M, weight, M)              # DCT 2D del kernel
    uh, uw = min(p, ph), min(p, pw)
    out = torch.einsum("ux,dcuv,vy->dcxy", Mh[:uh], coef[:, :, :uh, :uw], Mw[:uw])
    return out * math.sqrt(p / ph) * math.sqrt(p / pw)


if __name__ == "__main__":
    import torch.nn.functional as F
    torch.manual_seed(0)
    w = torch.randn(16, 1, 8, 8)
    assert torch.allclose(rescale_patch_kernel(w, (8, 8)), w, atol=1e-5)
    print("1) mismo tamaño de parche -> mismo kernel: OK")
    lo = F.interpolate(torch.randn(4, 1, 16, 16), size=(64, 64), mode="bicubic")     # imagen suave de 64x64
    ref = F.conv2d(lo, w, stride=8)
    for hw in [(32, 32), (96, 64), (128, 128)]:
        up = F.interpolate(lo, size=hw, mode="bicubic", antialias=hw[0] < 64)
        pt = (8 * hw[0] // 64, 8 * hw[1] // 64)
        tok = F.conv2d(up, rescale_patch_kernel(w, pt), stride=pt)
        assert tok.shape == ref.shape
        print(f"   {hw}: parche {pt}  error relativo de tokens {((tok - ref).norm() / ref.norm()).item():.3f}")
    print("2) imagen reescalada + kernel reescalado -> misma rejilla y tokens parecidos: OK")
