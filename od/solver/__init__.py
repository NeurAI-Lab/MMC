from od.solver import registry
from od.solver.sgd_optimizer import SGD_optimizer
from od.solver.adam_optimizer import ADAM_optimizer,New_ADAM_optimizer
__all__ = ['make_optimizer', 'SGD_optimizer', 'ADAM_optimizer','New_ADAM_optimizer']

def make_optimizer(cfg, model, lr=None):
    return registry.SOLVERS[cfg.SOLVER.NAME](cfg, model, lr)

