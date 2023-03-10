from torch.optim.lr_scheduler import LambdaLR
from pytorch_lightning.callbacks import Callback


class FileLR(LambdaLR):
    def __init__(self, optimizer, path='config/learningRate', *args, **kwargs):
        self.path = path
        self.optimizer_initial_lr = optimizer.defaults['lr']
        super(FileLR, self).__init__(optimizer, self.lr_lambda, *args, **kwargs)

    def lr_lambda(self, epoch):
        with open(self.path, 'r') as f:
            lr = float(f.read())
        return lr / self.optimizer_initial_lr


class FileLRCallback(Callback):
    def __init__(self, path='config/learningRate'):
        self.path = path

    def on_fit_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"):
        for config in trainer.lr_scheduler_configs:
            optimizer = config.scheduler.optimizer
            config.scheduler = FileLR(optimizer, self.path)
            config.interval = 'step'
            config.frequency = 1
            config.reduce_on_plateau = False