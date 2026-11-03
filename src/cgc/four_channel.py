"""Four independent E0 scalar channels (a 120-parameter reference wrapper).

This wrapper is new packaging code, not the qualified historical E0 source.
"""
import torch
from torch import nn
from .ecsg_e0 import ECSGE0Cell

class CGC120(nn.Module):
    def __init__(self, initial_weights: torch.Tensor):
        super().__init__()
        if (not isinstance(initial_weights, torch.Tensor)
            or initial_weights.shape != (4, 5, 6)
            or initial_weights.dtype != torch.float64
            or not torch.isfinite(initial_weights).all()):
            raise ValueError('CGC120 expects finite float64 weights of shape [4,5,6]')
        self.channels = nn.ModuleList([ECSGE0Cell(initial_weights[i]) for i in range(4)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.stack([channel(x) for channel in self.channels], dim=-1)
