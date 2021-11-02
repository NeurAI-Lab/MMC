import torch
from od.solver import registry
from torch.optim.lr_scheduler import _LRScheduler

@registry.SCHEDULERS.register('WarmupMultiScaleLR')
def WarmupMultiScaleLR(cfg, optimizer, milestones):
    return WarmupMultiScaleLR(optimizer=optimizer,
                              lr_steps=cfg.SOLVER.LR_STEPS,
                              lr_scales=cfg.MODEL.LR_SCALES,
                              weight_decay=cfg.SOLVER.WEIGHT_DECAY,
                              batch_size=cfg.SOLVER.BATCH_SIZE)

class WarmupMultiScaleLR(_LRScheduler):
    """Sets the learning rate of each parameter group to the initial lr
    decayed by 'lr_scales' at every 'lr_steps' iterations. The lr is also divided by batch size and
    the weight decay multiplied by the batch size. (From Yolo paper)
    """
    def __init__(self, optimizer, lr_steps, lr_scales, weight_decay, batch_size, last_epoch=-1):
        self.lr_scales = lr_scales
        self.lr_steps = lr_steps
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        if(self.last_epoch == 0):
            for param_group in self.optimizer.param_groups:
                param_group['weight_decay'] = self.weight_decay * self.batch_size

        scaled_lrs = [base_lr for base_lr in self.base_lrs]
        for i in range(len(self.lr_steps)):
            scale = self.lr_scales[i] if i < len(self.lr_scales) else 1
            if self.last_epoch >= self.lr_steps[i]:
                scaled_lrs = [base_lr * scale for base_lr in self.base_lrs]
                if self.last_epoch == self.lr_steps[i]:
                    break
            else:
                break
        return [
            base_lr/self.batch_size for base_lr in scaled_lrs
        ]