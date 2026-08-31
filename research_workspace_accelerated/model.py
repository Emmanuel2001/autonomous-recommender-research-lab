"""PyTorch Factorization Machine for CPU, Apple MPS, or NVIDIA CUDA. AGENT-EDITABLE."""
# requires: torch
from __future__ import annotations

import numpy as np
import torch
from torch import nn


class TorchFM(nn.Module):
    """Second-order FM with the same mathematical forward pass as the NumPy seed."""

    def __init__(self, dim: int, k: int = 16, seed: int = 0):
        super().__init__()
        # Initialise on CPU with NumPy so the seed has a framework-independent definition;
        # moving to MPS/CUDA afterwards cannot change the starting values.
        rng = np.random.default_rng(seed)
        v = rng.normal(0, 0.01, (dim, int(k))).astype(np.float32)
        self.V = nn.Parameter(torch.from_numpy(v))
        self.W = nn.Parameter(torch.zeros(dim, dtype=torch.float32))
        self.b = nn.Parameter(torch.zeros((), dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e = self.V[x]
        summed = e.sum(dim=1)
        interaction = 0.5 * ((summed * summed).sum(dim=1)
                             - (e * e).sum(dim=(1, 2)))
        return self.b + self.W[x].sum(dim=1) + interaction

    @torch.no_grad()
    def predict_numpy(self, x: np.ndarray, device: str, batch: int = 200_000) -> np.ndarray:
        self.eval()
        out = []
        for i in range(0, len(x), batch):
            xb = torch.as_tensor(x[i:i + batch], dtype=torch.long, device=device)
            out.append(self(xb).detach().cpu().numpy())
        return np.concatenate(out) if out else np.zeros(0, dtype=np.float32)
