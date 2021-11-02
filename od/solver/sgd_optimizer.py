import torch
from od.solver import registry
@registry.SOLVERS.register('SGD_optimizer')
def SGD_optimizer(cfg, model, lr=None):
    lr = cfg.SOLVER.BASE_LR if lr is None else lr
    return torch.optim.SGD(model.parameters(), lr=lr, momentum=cfg.SOLVER.MOMENTUM, weight_decay=cfg.SOLVER.WEIGHT_DECAY)