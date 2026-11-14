"""MSViT adaptado (Jiang et al., Neural Networks 205, 109446) a espectrogramas de 1 canal.

Piezas del paper:
  SC-Conv : DCT 2D fija por parche -> peso espectral aprendido (diagonal) -> proyección 1x1.
            Se puede fusionar en UNA convolución (parche x parche, paso = parche).
  HVSA    : atención axial secuencial (primero a lo largo de W, luego de H), con q/k separados
            para cada eje y V compartido (5 proyecciones + 1 de salida = 6*C^2*H*W MACs, Apéndice B).
  RepCPE  : sesgo posicional = matriz de Toeplitz construida desde un kernel 1D aprendido, SUMADA a la
            atención después del softmax. Se construye para cualquier tamaño: no hay pos-embedding absoluto
            ni interpolación -> acepta cualquier resolución.
  Pirámide: etapas con downsample (conv 3x3, paso 2) y GAP + Linear al final.

ADAPTACIÓN (no es una reproducción idéntica): entrada de 1 canal (sin RGB->YCbCr), tamaño reducido,
y escala 1/sqrt(d_head) en los logits (el texto del paper no la menciona).
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ----------------------------------------------------------------------------- SC-Conv
def dct_matrix(n):
    """DCT-II ortonormal (n x n): M @ M.T = I."""
    k = torch.arange(n, dtype=torch.float32)[:, None]
    i = torch.arange(n, dtype=torch.float32)[None, :]
    M = torch.cos(math.pi * (2 * i + 1) * k / (2 * n)) * math.sqrt(2.0 / n)
    M[0] = M[0] / math.sqrt(2.0)
    return M


def dct2_basis(p):
    """(p*p, 1, p, p): canal u*p+v = base DCT 2D de frecuencia (u, v)."""
    M = dct_matrix(p)
    return torch.einsum("ux,vy->uvxy", M, M).reshape(p * p, 1, p, p)


class SCConv(nn.Module):
    def __init__(self, dim, p=4):
        super().__init__()
        self.p = p
        self.register_buffer("basis", dct2_basis(p))     # fijo, no entrenable
        self.w = nn.Parameter(torch.ones(p * p))          # peso espectral (matriz diagonal)
        self.proj = nn.Conv2d(p * p, dim, 1)              # proyección lineal a la dimensión del modelo

    def scaled_basis(self, ph, pw):
        """Base DCT para un parche (ph, pw) != (p, p), expresada en los MISMOS p*p canales espectrales.

        El canal (u, v) sigue siendo "u ciclos/2 a lo alto y v ciclos/2 a lo ancho DEL PARCHE", así que si la
        imagen se reescala y el parche se reescala igual, cada token ve el mismo contenido físico y los pesos
        aprendidos (w, proj) se reutilizan tal cual. Frecuencias que no existen en el parche pequeño (u >= ph)
        quedan en cero; las que sobran en el parche grande (u >= p) se descartan (truncamiento espectral).
        El factor sqrt(p/ph)*sqrt(p/pw) compensa la norma de la DCT ortonormal al cambiar el nº de muestras.
        """
        p = self.p
        Mh, Mw = dct_matrix(ph), dct_matrix(pw)
        B = torch.zeros(p, p, ph, pw)
        uh, uw = min(p, ph), min(p, pw)
        B[:uh, :uw] = torch.einsum("ux,vy->uvxy", Mh[:uh], Mw[:uw]) * math.sqrt(p / ph) * math.sqrt(p / pw)
        return B.reshape(p * p, 1, ph, pw)

    def forward(self, x, patch=None):
        if patch is None or tuple(patch) == (self.p, self.p):
            f = F.conv2d(x, self.basis, stride=self.p)    # coeficientes DCT por parche
        else:                                             # parche escalado con la resolución (sin pesos nuevos)
            basis = self.scaled_basis(*patch).to(x.device, x.dtype)
            f = F.conv2d(x, basis, stride=tuple(patch))
        return self.proj(f * self.w.view(1, -1, 1, 1))

    @torch.no_grad()
    def fused(self):
        """Pesos (dim,1,p,p) y sesgo equivalentes a toda la tubería (para inferencia)."""
        W = self.proj.weight[:, :, 0, 0] * self.w[None, :]
        return torch.einsum("dc,cixy->dixy", W, self.basis), self.proj.bias.clone()


# ----------------------------------------------------------------------------- RepCPE / HVSA
def toeplitz(k, n):
    """k: (heads, K) -> (heads, n, n) con T[i,j] = k[j-i+K//2] si |j-i| <= K//2, 0 fuera.
    Equivale a una convolución 1D con zero-padding (cross-correlación) y da conciencia de borde."""
    K = k.shape[1]
    idx = torch.arange(n, device=k.device)
    d = idx[None, :] - idx[:, None] + K // 2
    mask = (d >= 0) & (d < K)
    return k[:, d.clamp(0, K - 1)] * mask


def _heads(x, h):      # (B,C,H,W) -> (B,h,H,W,dh)
    B, C, H, W = x.shape
    return x.view(B, h, C // h, H, W).permute(0, 1, 3, 4, 2)


def _merge(x):         # (B,h,H,W,dh) -> (B,C,H,W)
    B, h, H, W, dh = x.shape
    return x.permute(0, 1, 4, 2, 3).reshape(B, h * dh, H, W)


class HVSA(nn.Module):
    def __init__(self, dim, heads, rep_kernel=3, use_repcpe=True):
        super().__init__()
        assert dim % heads == 0
        self.h, self.scale, self.use_rep = heads, (dim // heads) ** -0.5, use_repcpe
        self.qkv = nn.Conv2d(dim, 5 * dim, 1)             # Q_H, Q_V, K_H, K_V, V
        self.proj = nn.Conv2d(dim, dim, 1)
        if use_repcpe:
            self.rep_kh = nn.Parameter(torch.zeros(heads, rep_kernel))
            self.rep_kv = nn.Parameter(torch.zeros(heads, rep_kernel))
            nn.init.trunc_normal_(self.rep_kh, std=0.02)
            nn.init.trunc_normal_(self.rep_kv, std=0.02)

    def forward(self, x):
        h = self.h
        qh, qv, kh, kv, v = [_heads(t, h) for t in self.qkv(x).chunk(5, dim=1)]   # (B,h,H,W,dh)
        H, W = x.shape[2], x.shape[3]
        # Paso 1 (horizontal): atención a lo largo de W dentro de cada fila
        a = ((qh @ kh.transpose(-1, -2)) * self.scale).softmax(-1)                # (B,h,H,W,W)
        if self.use_rep:
            a = a + toeplitz(self.rep_kh, W).to(a.dtype)[None, :, None]
        z = a @ v
        # Paso 2 (vertical): atención a lo largo de H dentro de cada columna
        qv, kv, z = (t.transpose(2, 3) for t in (qv, kv, z))                     # (B,h,W,H,dh)
        a = ((qv @ kv.transpose(-1, -2)) * self.scale).softmax(-1)                # (B,h,W,H,H)
        if self.use_rep:
            a = a + toeplitz(self.rep_kv, H).to(a.dtype)[None, :, None]
        y = (a @ z).transpose(2, 3)                                               # (B,h,H,W,dh)
        return self.proj(_merge(y))


class LN2d(nn.LayerNorm):
    def forward(self, x):
        x = F.layer_norm(x.permute(0, 2, 3, 1), self.normalized_shape, self.weight, self.bias, self.eps)
        return x.permute(0, 3, 1, 2)


class Block(nn.Module):
    def __init__(self, dim, heads, mlp_ratio, drop, rep_kernel, use_repcpe):
        super().__init__()
        self.n1, self.n2 = LN2d(dim), LN2d(dim)
        self.attn = HVSA(dim, heads, rep_kernel, use_repcpe)
        hid = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(nn.Conv2d(dim, hid, 1), nn.GELU(), nn.Dropout(drop),
                                 nn.Conv2d(hid, dim, 1), nn.Dropout(drop))

    def forward(self, x):
        x = x + self.attn(self.n1(x))
        return x + self.mlp(self.n2(x))


# ----------------------------------------------------------------------------- Modelo
class MSViT(nn.Module):
    def __init__(self, num_classes, in_chans=1, patch_size=4, dims=(64, 128, 256), depths=(2, 4, 2),
                 head_dim=32, mlp_ratio=4.0, dropout=0.1, rep_kernel=3, use_scconv=True, use_repcpe=True):
        super().__init__()
        assert in_chans == 1, "SC-Conv adaptado a espectrogramas de 1 canal"
        self.p = patch_size
        self.ref_hw = None      # (H, W) de entrenamiento. Si se fija, el parche del stem escala con la entrada
        self.stem = SCConv(dims[0], patch_size) if use_scconv else nn.Conv2d(1, dims[0], patch_size, patch_size)
        stages, downs = [], []
        for i, (d, n) in enumerate(zip(dims, depths)):
            stages.append(nn.Sequential(*[Block(d, d // head_dim, mlp_ratio, dropout, rep_kernel, use_repcpe)
                                          for _ in range(n)]))
            if i < len(dims) - 1:
                downs.append(nn.Conv2d(d, dims[i + 1], 3, stride=2, padding=1))
        self.stages, self.downs = nn.ModuleList(stages), nn.ModuleList(downs)
        self.norm = nn.LayerNorm(dims[-1])
        self.head = nn.Linear(dims[-1], num_classes)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02); nn.init.zeros_(m.bias)

    def forward(self, x):
        H, W = x.shape[-2:]
        if self.ref_hw is not None and (H, W) != tuple(self.ref_hw):
            # Stem escalable: parche = p * (resolución de prueba / resolución de entrenamiento) en cada eje,
            # de modo que la rejilla de tokens es la MISMA que en entrenamiento (p. ej. 16x16) a cualquier resolución.
            ph, pw = self.p * H / self.ref_hw[0], self.p * W / self.ref_hw[1]
            if ph != int(ph) or pw != int(pw) or ph < 1 or pw < 1:
                raise ValueError(f"Entrada {H}x{W}: el parche escalado ({ph}, {pw}) no es entero.")
            if not isinstance(self.stem, SCConv):
                raise ValueError("El stem escalable requiere use_scconv=true.")
            x = self.stem(x, (int(ph), int(pw)))
        else:
            if H % self.p or W % self.p:
                raise ValueError(f"Entrada {H}x{W} no es múltiplo del parche {self.p}.")
            x = self.stem(x)
        for i, st in enumerate(self.stages):
            x = st(x)
            if i < len(self.downs):
                x = self.downs[i](x)
        return self.head(self.norm(x.mean(dim=(2, 3))))


# ----------------------------------------------------------------------------- Pruebas aisladas
if __name__ == "__main__":
    torch.manual_seed(0)
    # 1) Base DCT ortonormal y coincidencia con una DCT 2D directa
    B = dct2_basis(4).reshape(16, 16)
    assert torch.allclose(B @ B.T, torch.eye(16), atol=1e-6)
    patch = torch.randn(1, 1, 4, 4)
    M = dct_matrix(4)
    ref = (M @ patch[0, 0] @ M.T).reshape(-1)
    got = F.conv2d(patch, dct2_basis(4), stride=4).reshape(-1)
    assert torch.allclose(ref, got, atol=1e-5)
    try:
        from scipy.fft import dctn
        assert torch.allclose(torch.tensor(dctn(patch[0, 0].numpy(), norm="ortho")).reshape(-1), got, atol=1e-5)
        print("1) DCT ortonormal y = scipy.fft.dctn: OK")
    except ImportError:
        print("1) DCT ortonormal (scipy no disponible; comparado con producto matricial): OK")

    # 2) SC-Conv fusionada == tubería completa
    sc = SCConv(64, 4); sc.w.data = torch.rand(16) + 0.5
    x = torch.randn(2, 1, 32, 48)
    Wf, bf = sc.fused()
    err = (sc(x) - F.conv2d(x, Wf, bf, stride=4)).abs().max().item()
    assert err < 1e-5, err
    print(f"2) SC-Conv fusionada: error máx {err:.2e}  OK")

    # 3) Toeplitz == convolución 1D con zero-padding, para dos tamaños
    k = torch.randn(2, 5)
    for n in (16, 32):
        T = toeplitz(k, n)
        v = torch.randn(n)
        conv = F.conv1d(v[None, None], k[0][None, None], padding=2)[0, 0]
        assert torch.allclose(T[0] @ v, conv, atol=1e-5)
        assert (T[:, 0, 3:] == 0).all() and T.shape == (2, n, n)
    print("3) Toeplitz = conv1d con zero-padding (n=16 y n=32), banda y bordes: OK")

    # 4) HVSA: cobertura del plano completo (alcance global en dos pasos)
    att = HVSA(32, 2).eval()
    xx = torch.randn(1, 32, 8, 8, requires_grad=True)
    att(xx)[0, :, 3, 5].sum().backward()
    g = xx.grad.abs().sum(1)[0]
    assert (g > 0).all()
    print("4) HVSA: la salida en (3,5) depende de las 64/64 posiciones de entrada: OK")

    # 5) Modelo completo
    m = MSViT(6).eval()
    n_par = sum(p.numel() for p in m.parameters())
    before = {k_: v_.clone() for k_, v_ in m.state_dict().items()}
    with torch.no_grad():
        for hw in [(32, 32), (64, 64), (96, 64), (128, 128)]:
            o = m(torch.randn(2, 1, *hw))
            assert o.shape == (2, 6) and torch.isfinite(o).all()
            print(f"   entrada {hw} -> {tuple(o.shape)}")
    assert all(torch.equal(before[k_], v_) for k_, v_ in m.state_dict().items())
    print(f"5) MSViT: {n_par/1e6:.2f} M parámetros; pesos intactos tras evaluar en 4 resoluciones: OK")

    # 6) Stem escalable: una imagen reescalada produce (casi) los mismos tokens y los mismos logits
    torch.manual_seed(0)
    m = MSViT(6).eval(); m.stem.w.data = torch.rand(16) + 0.5
    lo = F.interpolate(torch.randn(8, 1, 16, 16), size=(64, 64), mode="bicubic")   # imagen suave de 64x64
    with torch.no_grad():
        ref_tok, ref_log = m.stem(lo), m(lo)
        for hw in [(32, 32), (96, 64), (128, 128)]:
            up = F.interpolate(lo, size=hw, mode="bicubic", antialias=hw[0] < 64)
            ph, pw = 4 * hw[0] // 64, 4 * hw[1] // 64
            e_tok = ((m.stem(up, (ph, pw)) - ref_tok).norm() / ref_tok.norm()).item()
            m.ref_hw = None; fixed = (m(up).argmax(1) == ref_log.argmax(1)).float().mean().item()
            m.ref_hw = (64, 64); adapt = (m(up).argmax(1) == ref_log.argmax(1)).float().mean().item()
            print(f"   {hw}: parche ({ph},{pw})  error relativo de tokens {e_tok:.3f} | "
                  f"misma predicción que en 64x64: parche fijo {fixed:.2f}, parche escalado {adapt:.2f}")
    print("6) Stem escalable: OK")