import torch
from od.solver import registry

def add_weight_decay(model, weight_decay=1e-5, skip_list=()):
    decay = []
    no_decay = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue  # frozen weights
        if len(param.shape) == 1 or name.endswith(".bias") or name in skip_list:
            no_decay.append(param)
        else:
            decay.append(param)
    return [
        {'params': no_decay, 'weight_decay': 0.},
        {'params': decay, 'weight_decay': weight_decay}]

@registry.SOLVERS.register('Transformer')
def New_ADAM_optimizer(cfg, model, lr=None):
    if cfg.SOLVER.WEIGHT_DECAY==0.0:
        weight_decay = .05
    else:
        weight_decay =  cfg.SOLVER.WEIGHT_DECAY
    parameters = add_weight_decay(model, weight_decay)
    opt_args={'lr':lr, 'weight_decay': 0.0, 'eps': 1e-08}
    return torch.optim.AdamW(parameters, **opt_args)


@registry.SOLVERS.register('ADAM_optimizer')
def ADAM_optimizer(cfg, model, lr=None):
    lr = cfg.SOLVER.BASE_LR if lr is None else lr
    return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=cfg.SOLVER.WEIGHT_DECAY)