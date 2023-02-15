import torch.nn as nn
import torch.nn.functional as F
from  torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
import pytorch_lightning as pl

from module.layer.InitLayer import PositionalEncoding, Patchfy, AppendClassToken
from module.layer.TransformerEncoder import TransformerEncoder

from torchmetrics.functional import accuracy
from einops import rearrange, unpack
from einops.layers.torch import Rearrange


class VIT(pl.LightningModule):
  def __init__(self, 
    input_shape:tuple, 
    num_classes:int, 
    patch_size:int, 
    num_repeat:int, 
    num_head:int, 
    learning_rate:float,
    dropout:float=0.1
    ):
    
    super().__init__()
    self.save_hyperparameters()

    # parse input_size
    assert len(input_shape) == 3, f'input_shape should have 3 dimensions, Channel, Height, Width.'
    input_channel, input_height, input_width = input_shape 

    # calculate token_dim & num_token
    token_dim = patch_size * patch_size * input_channel
    assert input_height % patch_size == 0, "input_height is indivisible with patch_size"
    num_token = (input_height // patch_size) * (input_width // patch_size) + 1 # CLASS token

    # layers
    self.Patchfy = Patchfy(patch_size)
    self.PositionalEncoding = PositionalEncoding(token_dim, dropout)
    self.AppendClassToken = AppendClassToken(token_dim)

    TES = nn.Sequential()
    for i in range(num_repeat):
      TES.add_module(f'TE{i+1}', TransformerEncoder(num_token, token_dim, num_head, dropout))
    self.TES = TES

    self.ClassifyHead = nn.Linear(token_dim, num_classes)

    # init parameters reccursively
    self.apply(self._init_parameters)

  def _init_parameters(self, module):
    if isinstance(module, nn.Linear) or isinstance(module, nn.LayerNorm):
      nn.init.xavier_normal_(module.weight)
      if module.bias is not None:
        nn.init.zeros_(module.bias)

  ##############################################################################

  def _shared_step(self, batch, batch_idx):
    x, label = batch
    batch_size = x.shape[0]

    # 1. Initialize
    x = self.Patchfy(x)
    x, ps = self.AppendClassToken(x)
    x = self.PositionalEncoding(x)

    # 2. TransformerEncoders
    x = self.TES(x)

    # 3. Classify classes
    class_token, _ = unpack(x, ps, 'N * E')
    score = self.ClassifyHead(class_token)

    # 4. calculate loss, accuracy
    loss = F.cross_entropy(score, label)
    acc = accuracy(score, label, task="multiclass", num_classes=self.hparams.num_classes)
    return loss, acc

  def training_step(self, batch, batch_idx):
    loss, acc = self._shared_step(batch, batch_idx)
    metrics = {"train_loss":loss, "train_accuracy":acc}
    self.log_dict(metrics)
    return loss

  def validation_step(self, batch, batch_idx):
    loss, acc = self._shared_step(batch, batch_idx)
    metrics = {"val_loss":loss, "val_accuracy":acc}
    self.log_dict(metrics)

  ##############################################################################

  def configure_optimizers(self):
    optimizer = Adam(self.parameters(), lr=self.hparams.learning_rate)
    return {
      "optimizer": optimizer,
      "lr_scheduler": {
        "scheduler": ReduceLROnPlateau(optimizer, mode='min'),
        "monitor": "val_loss",
      },
    }

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, threshold=0.0001, threshold_mode='rel', cooldown=0, min_lr=0, eps=1e-08, verbose=False)