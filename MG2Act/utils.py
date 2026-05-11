import torch
import torch.nn as nn


class LayerNorm(nn.Module):
    """Improved layer normalization"""
    def __init__(self, features, eps=1e-6):
        super().__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


class ConvPooler(nn.Module):
    """Convolutional pooler"""
    def __init__(self, indim, outdim, kernel_size, stride):
        super().__init__()
        self.conv = nn.Conv1d(indim, outdim, kernel_size, stride, padding=kernel_size//2)
        self.relu = nn.ReLU()
        self.norm = LayerNorm(outdim)

    def forward(self, x):
        """
        Input: [batch_size, seq_len, indim]
        Output: [batch_size, seq_len', outdim]
        """
        xt = x.transpose(1, 2)  
        xc = self.conv(xt)  
        xc = self.relu(xc)
        y = xc.transpose(1, 2)  
        y = self.norm(y)
        return y
