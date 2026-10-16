"""Construcción de tensores (espectrogramas) por resolución, a partir de I/Q + índices de partición."""
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader

from preprocessing.stft_generator import compute_spectrograms, resolution_shape


def build_dataset(subset, indices, res, stft_cfg, device="cpu", batch=512):
    iq, y = subset["X"], subset["y"]
    xs = []
    for i in range(0, len(indices), batch):
        b = torch.from_numpy(iq[indices[i:i + batch]]).to(device)
        s = compute_spectrograms(b, res.n_fft, res.win_len, res.hop, window=stft_cfg.window,
                                 log_scale=stft_cfg.log_scale, normalize=(stft_cfg.normalize == "per_sample"))
        xs.append(s.unsqueeze(1).cpu())
    return TensorDataset(torch.cat(xs), torch.from_numpy(y[indices]).long())


def make_loader(ds, batch_size, shuffle, num_workers=0, pin=False):
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=pin)


def res_hw(res, signal_len):
    return resolution_shape(signal_len, res.n_fft, res.hop)
