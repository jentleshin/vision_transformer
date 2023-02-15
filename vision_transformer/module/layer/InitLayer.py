import torch
import torch.nn as nn
from einops import rearrange, pack

class PositionalEncoding(nn.Module):
  def __init__(self, token_dim: int, dropout: float=0.1, max_len: int=5000):
    super().__init__()
    assert token_dim % 2 == 0

    pe = torch.zeros(1, max_len, token_dim)
    pe[:,:,0::2] = pe[:,:,1::2] = torch.arange(start=0, end=token_dim, step=2)
    pe = torch.pow(10000, pe / (-1 * token_dim))
    pe = pe * torch.arange(max_len).view(-1,1)
    pe[:,:,0::2], pe[:,:,1::2] = torch.sin(pe[:,:,0::2]), torch.cos(pe[:,:,1::2])

    self.token_dim = token_dim
    self.register_buffer('pe', pe)
    self.dropout = nn.Dropout(p=dropout)

  def forward(self, x):
    N, H, E = x.shape
    assert E == self.token_dim, "input shape does not match with initial 'token_dim'"
    
    x += self.pe[:,:H,:]
    y = self.dropout(x)
    return y

################################################################################

class Patchfy(nn.Module):
  def __init__(self, patch_size: int):
    super().__init__()
    self.patch_size = patch_size

  def forward(self,x):
    P = self.patch_size
    H, W = x.shape[-2], x.shape[-1]
    assert H % P == 0, "Height is not divisible with 'patch_size'"
    assert W % P == 0, "Width is not divisible with 'patch_size'"
 
    y = rearrange(x, "N C (nH P1) (nW P2) -> N (nH nW) (P1 P2 C)", P1=P, P2=P)
    return y

################################################################################

class AppendClassToken(nn.Module):
  def __init__(self, token_dim: int):
    super().__init__()
    self.token_dim = token_dim
    self.class_token = nn.UninitializedParameter()

  def forward(self, x):
    N, T, E = x.shape
    assert E == self.token_dim, "input shape does not match with initial 'token_dim'"
    
    w = torch.empty(N, E)
    nn.init.kaiming_normal_(w)
    self.class_token = nn.Parameter(w)

    y, ps = pack([self.class_token, x], "N * E")
    return y, ps
