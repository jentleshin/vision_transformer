import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from einops import rearrange
from einops.layers.torch import Rearrange

################################################################################

class MultiHeadAttention(nn.Module):
  def __init__(self, num_token: int, token_dim: int ,num_head: int, dropout: int):
    super().__init__()
    
    # T: num_token, E: token_dim, H: num_head, D: head_dim
    assert token_dim % num_head == 0 ,"token_dim is not divisible with num_head"
    self.T, E = num_token, token_dim
    self.H, self.D = num_head, token_dim // num_head

    # E = H*D 
    self.get_query = nn.Linear(E, E, bias= False)
    self.get_key = nn.Linear(E, E, bias= False)
    self.get_value = nn.Linear(E, E, bias= False)

    # dropout
    self.dropout = nn.Dropout(p=dropout)

  def forward(self, x):
    # input x: (N, T, E)
    N, _T, _E = x.shape
    T, H, D = self.T, self.H, self.D
    assert _T == T, "input shape does not match with the given 'num_token'"
    assert _E == H * D, "input shape does not match with the given 'token_dim'"

    # query, key, value: (N, T, H, D)
    query = self.get_query(x)
    query = rearrange(query, 'N T (H D) -> N H T D', H=H, D=D)
    
    key = self.get_key(x)
    key = rearrange(key, 'N T (H D) -> N H D T', H=H, D=D)
    
    value = self.get_value(x)
    value = rearrange(value, 'N T (H D) -> N H T D', H=H, D=D)
    
    # score, probability: (N H T D) x (N H D T) = (N H T T)
    norm_val = math.sqrt(D)
    score = torch.matmul(query, key) / norm_val
    prob = F.softmax(score, dim=3)
    
    # dropout?
    prob = self.dropout(prob)
    
    # weighted sum: (N H T T) x (N H T D) = (N H T D)
    wsum = torch.matmul(prob, value)
    wsum = rearrange(wsum, 'N H T D -> N T (H D)')
    return wsum

################################################################################

class TransformerEncoder(nn.Module):
  def __init__(self, num_token: int, token_dim: int ,num_head: int, dropout: int):
    super().__init__()

    # T: num_token, E: token_dim, H: num_head, D: head_dim
    assert token_dim % num_head == 0 ,"token_dim is not divisible with num_head"
    T, H, D = self.T, self.H, self.D = num_token, num_head, token_dim // num_head
    E = token_dim

    # MultiHeadAttention Module    
    self.MHA = nn.Sequential(
      nn.LayerNorm([T, E]),
      MultiHeadAttention(T, E, H, dropout)
    )

    # MultiLayerPerceptron Modue
    self.MLP = nn.Sequential(
      nn.LayerNorm([T, E]),
      Rearrange('N T E -> N (T E)'),
      nn.Linear(T*E, T*E),
      nn.GELU(),
      nn.Linear(T*E, T*E),
      Rearrange('N (T E) -> N T E', T=T, E=E),
    )

  def forward(self, x):
    y = self.MHA(x) + x
    z = self.MLP(y) + y
    return z