from od.solver import registry
from torch.optim.lr_scheduler import _LRScheduler
from od.solver.multi_step_scheduler import WarmupMultiStepLR
from od.solver.multi_scale_scheduler import WarmupMultiScaleLR
from od.solver.polynomial_scheduler import PolynomialLR
from od.solver.cosine_scheduler import CosineLR,NewCosineLR
__all__ = ['make_lr_scheduler', 'WarmupMultiStepLR', 'WarmupMultiScaleLR', 'PolynomialLR', 'CosineLR','NewCosineLR']

def make_lr_scheduler(cfg, optimizer, milestones=None):
    return registry.SCHEDULERS[cfg.SCHEDULER.TYPE](cfg, optimizer, milestones)






