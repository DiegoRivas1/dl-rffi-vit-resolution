"""STFT de señales I/Q complejas -> representación tiempo-frecuencia.

Resolución = parámetros de generación (n_fft, win_len, hop), NO un resize:
    F (alto)  = n_fft            bins de frecuencia (espectro de dos lados, fftshift)
    T (ancho) = signal_len//hop  frames de tiempo (ventanas centradas, padding reflejado)

Notas metodológicas:
  * La resolución frecuencial real la fija win_len (~1/win_len); si n_fft > win_len los
    bins extra solo interpolan (zero-padding). Por eso los presets usan win_len = n_fft.
  * La resolución temporal la fija hop.
"""
import torch
import torch.nn.functional as Fn


def resolution_shape(signal_len, n_fft, hop):
    return n_fft, signal_len // hop


def _window(name, n, device, dtype):
    if name == "hann":
        return torch.hann_window(n, periodic=True, device=device, dtype=dtype)
    if name in ("rect", "boxcar", None):
        return torch.ones(n, device=device, dtype=dtype)
    raise ValueError(f"ventana no soportada: {name}")


def compute_spectrograms(iq, n_fft, win_len, hop, window="hann",
                         log_scale=True, normalize=True, eps=1e-10):
    """iq: (B, N, 2) float -> (B, F, T) float32."""
    dec = 1
    if win_len > n_fft:
        # Ventana más larga que el nº de bins pedido: se calcula la FFT de win_len puntos y se toma 1 de cada
        # `dec` bins (el mismo espectro, muestreado más grueso). Solo para el control de ventana fija.
        if win_len % n_fft:
            raise ValueError("si win_len > n_fft, win_len debe ser múltiplo de n_fft")
        dec = win_len // n_fft
    iq = iq.float()
    B, N, _ = iq.shape
    pad = win_len // 2
    xr = Fn.pad(iq[..., 0].unsqueeze(1), (pad, pad), mode="reflect").squeeze(1)
    xi = Fn.pad(iq[..., 1].unsqueeze(1), (pad, pad), mode="reflect").squeeze(1)
    x = torch.complex(xr, xi)                                  # (B, N+2*pad)
    T = N // hop
    frames = x.unfold(-1, win_len, hop)[:, :T]                  # (B, T, win_len)
    w = _window(window, win_len, iq.device, iq.dtype)
    frames = frames * w
    spec = torch.fft.fft(frames, n=n_fft * dec, dim=-1)[..., ::dec]   # (B, T, n_fft)
    spec = torch.fft.fftshift(spec, dim=-1)
    power = spec.real ** 2 + spec.imag ** 2
    out = power.transpose(1, 2).contiguous()                    # (B, F, T)
    if log_scale:
        out = 10.0 * torch.log10(out + eps)
    if normalize:  # z-score por muestra (no depende de estadísticas globales)
        m = out.mean(dim=(1, 2), keepdim=True)
        s = out.std(dim=(1, 2), keepdim=True)
        out = (out - m) / (s + 1e-6)
    return out


if __name__ == "__main__":
    # Prueba aislada: formas y posición de un tono conocido.
    N = 256
    n = torch.arange(N, dtype=torch.float32)
    f0 = 0.125                                    # ciclos/muestra -> bin n_fft*f0 sobre el centro
    tone = torch.stack([torch.cos(2 * torch.pi * f0 * n), torch.sin(2 * torch.pi * f0 * n)], -1)[None]
    presets = {"R32x32": (32, 32, 8), "R64x64": (64, 64, 4), "R96x64": (96, 96, 4), "R128x128": (128, 128, 2)}
    for name, (nf, wl, hp) in presets.items():
        s = compute_spectrograms(tone, nf, wl, hp, normalize=False)
        assert tuple(s.shape[1:]) == resolution_shape(N, nf, hp), (name, s.shape)
        peak = int(s[0].mean(dim=1).argmax())
        expected = nf // 2 + round(f0 * nf)
        assert abs(peak - expected) <= 1, (name, peak, expected)
        print(f"{name}: shape={tuple(s.shape[1:])} pico_bin={peak} esperado={expected}  OK")
    # Ventana fija en 64: mismas formas y mismo pico; solo cambia el muestreo del espectro y del tiempo.
    for name, (nf, hp) in {"W64_R32x32": (32, 8), "W64_R96x64": (96, 4), "W64_R128x128": (128, 2)}.items():
        s = compute_spectrograms(tone, nf, 64, hp, normalize=False)
        assert tuple(s.shape[1:]) == resolution_shape(N, nf, hp), (name, s.shape)
        assert abs(int(s[0].mean(dim=1).argmax()) - (nf // 2 + round(f0 * nf))) <= 1, name
        print(f"{name}: shape={tuple(s.shape[1:])} (ventana 64)  OK")
    z = compute_spectrograms(torch.randn(4, N, 2), 64, 64, 4)
    assert abs(float(z.mean())) < 1e-4 and abs(float(z.std(dim=(1, 2)).mean()) - 1) < 1e-3
    print("normalización por muestra OK")
