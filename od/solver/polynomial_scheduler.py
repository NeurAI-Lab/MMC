import torch
from od.solver import registry
from torch.optim.lr_scheduler import _LRScheduler

@registry.SCHEDULERS.register('PolynomialLR')
def PolynomialLR(cfg, optimizer, milestones):
    return PolynomialLR(optimizer=optimizer,
                        batch_size=cfg.SOLVER.BATCH_SIZE,
                        max_iter=cfg.SOLVER.MAX_ITER)

class PolynomialLR(_LRScheduler):
    """Polynomial decay of learning rate
        :param init_lr is base learning rate
        :param iter is a current iteration
        :param lr_decay_iter how frequently decay occurs, default is 1
        :param max_iter is number of maximum iterations
        :param power is a polymomial power
    """
    def __init__(self, optimizer, batch_size, max_iter, lr_decay_iter=1, power=0.9, last_epoch=-1):
        self.lr_decay_iter = lr_decay_iter
        self.max_iter = max_iter
        self.power = power
        self.batch_size = batch_size
        super().__init__(optimizer, last_epoch)

    def get_lr(self):

        if self.last_epoch % self.lr_decay_iter or self.last_epoch > self.max_iter:
            return [base_lr / self.batch_size for base_lr in self.base_lrs]

        poly_factor = (1 - self.last_epoch/self.max_iter)**self.power
        scaled_lrs = [base_lr * poly_factor for base_lr in self.base_lrs]
        return [
            base_lr/self.batch_size for base_lr in scaled_lrs
        ]