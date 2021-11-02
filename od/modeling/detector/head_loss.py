import os
import torch
import torch.nn as nn
from od.modeling.head.centernet.CenterNetLossCalculator import CtdetLoss
from od.modeling.head.ssd.loss import MultiBoxLoss

def head_loss(cfg, features, targets):
    centernet_loss = CtdetLoss(cfg)
    loss_stats = centernet_loss(features, targets)
    return loss_stats


def ssd_loss(cfg, features, targets):
    ssd_loss = MultiBoxLoss(neg_pos_ratio=cfg.MODEL.NEG_POS_RATIO, num_classes=cfg.MODEL.NUM_CLASSES, focal_loss=cfg.MODEL.HEAD.FOCAL_LOSS)
    loss_stats = ssd_loss(features[0], features[1], targets['labels'], targets['boxes'])
    return loss_stats