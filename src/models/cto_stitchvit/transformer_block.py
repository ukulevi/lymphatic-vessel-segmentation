import torch
from torch import nn
from torch.nn import functional as F

class TransformerBlock(nn.Module):
    def __init__(self, in_channels, num_heads, with_ffn=True, ffn_dim=2048, dropout=0.1):
        super(TransformerBlock, self).__init__()
        self.attn = nn.MultiheadAttention(in_channels, num_heads, dropout=dropout)
        self.norm1 = nn.LayerNorm(in_channels)
        self.norm2 = nn.LayerNorm(in_channels)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        
        self.with_ffn = with_ffn
        if self.with_ffn:
            self.ffn = nn.Sequential(
                nn.Linear(in_channels, ffn_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
                nn.Linear(ffn_dim, in_channels)
            )

    def forward(self, x):
        x = x.permute(2, 0, 1) # HxW, B, C
        x_ = self.norm1(x)
        x_ = self.attn(x_, x_, x_)[0]
        x = x + self.dropout1(x_)
        
        if self.with_ffn:
            x_ = self.norm2(x)
            x_ = self.ffn(x_)
            x = x + self.dropout2(x_)
            
        x = x.permute(1, 2, 0) # B, C, HxW
        return x
