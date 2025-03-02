import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import constant_, xavier_uniform_

from ultralytics.nn.modules.conv import Conv

__all__ = (
    #"TransformerEncoderLayer",
    "TransformerLayer",
    "TransformerBlock",
    #"MLPBlock",
    #"LayerNorm2d",
    #"AIFI",
    #"DeformableTransformerDecoder",
    #"DeformableTransformerDecoderLayer",
    #"MSDeformAttn",
    #"MLP",
)



class TransformerLayer(nn.Module):
    """Transformer Layer"""

    def __init__(self, c, num_heads):
        super().__init__()
        self.q = nn.Linear(c, c, bias=False)
        self.k = nn.Linear(c, c, bias=False)
        self.v = nn.Linear(c, c, bias=False)
        self.ma = nn.MultiheadAttention(embed_dim=c, num_heads=num_heads)  #多头注意力机制
        self.fc1 = nn.Linear(c, c, bias=False)
        self.fc2 = nn.Linear(c, c, bias=False)

    def forward(self, x):
        x = self.ma(self.q(x), self.k(x), self.v(x))[0] + x
        return self.fc2(self.fc1(x)) + x

class TransformerBlock(nn.Module):
    """视觉注意力转换"""

    def __init__(self, c1, c2, num_heads, num_layers):
        super().__init__()
        self.conv = None
        if c1 != c2:
            self.conv = Conv(c1, c2)
        self.linear = nn.Linear(c2, c2)
        self.tr = nn.Sequential(*(TransformerLayer(c2, num_heads) for _ in range(num_layers)))
        self.c2 = c2

    def forward(self, x):
        if self.conv is not None:
            x = self.conv(x)
        b, _, w, h = x.shape
        p = x.flatten(2).permute(2, 0, 1)
        return self.tr(p + self.linear(p)).permute(1, 2, 0).reshape(b, self.c2, w, h)
