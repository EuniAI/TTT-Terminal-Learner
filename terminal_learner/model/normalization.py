"""RMS normalization shared by decoder blocks and Qwen3 Q/K heads."""
import torch
from torch import nn

class RMSNorm(nn.Module):
    def __init__(self, width, eps):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(width))
        self.eps = eps

    def forward(self, x):
        work = x if x.dtype == torch.float64 else x.float()
        return (work * torch.rsqrt(work.square().mean(-1, keepdim=True) + self.eps)).to(x.dtype) * self.weight


