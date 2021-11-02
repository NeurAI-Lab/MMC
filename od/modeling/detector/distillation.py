import torch
import math
import numpy as np
import torch.nn.functional as F
from torch import nn
#from pytorch_msssim import ssim, ms_ssim, SSIM, MS_SSIM
from od.modeling.head.centernet.utils import _sigmoid
from od.modeling.head.centernet.utils import _tranpose_and_gather_feat

criterion_MSE = nn.MSELoss(reduction='mean')

def at(x):
    return F.normalize(x.pow(2).mean(1).view(x.size(0), -1))

def collate_loss(cfg, clreg_loss, kd_loss, loss_dict, teacher=True):

    if teacher:
        str = "tch"
    else:
        str = "st"
    loss_dict[str +'_cl_loss'] = clreg_loss[0]
    loss_dict[str +'_reg_loss'] = clreg_loss[1]
    loss_dict['loss'] = 0

    if cfg.KD.ENABLE_DML == True:
        if 'KL' in cfg.KD.DISTILL_TYPE:
            loss_dict[str +'_kl_loss'] = kd_loss['distill_kl']
        if 'L2' in cfg.KD.DISTILL_TYPE:
            loss_dict[str +'_l2_loss'] = kd_loss['distill_l2']
        if 'AT' in cfg.KD.DISTILL_TYPE:
            loss_dict[str +'at_loss'] = kd_loss['distill_at']
        if 'L2_B' in cfg.KD.DISTILL_TYPE:
            loss_dict[str +'_l2b_loss'] = kd_loss['distill_l2_b']
        loss_dict['loss'] += kd_loss['loss']

    loss_dict['loss'] += clreg_loss[0] + clreg_loss[1]


def KD_detector(cfg, features_bt, features_bs, features_ht, features_hs, targets, loss_dict):

    loss = 0
    distill_type = cfg.KD.DISTILL_TYPE
    # attention loss at backbone
    if 'AT' in distill_type:
        loss_dict['distill_at'] = 0
        distill = distillation_loss('AT', features_bt, features_bs, at_layers=cfg.KD.DISTILL_AT_LAYERS)
        distill_loss = distill.at_loss()
        loss_dict['distill_at'] = distill_loss
        loss_dict['loss'] += cfg.KD.LOSS_WEIGHTS['at'] * loss_dict['distill_at']

    # KL loss for classifier
    if 'KL' in distill_type:
        loss_dict['distill_kl'] = 0
        distill = distillation_loss('KL', features_ht, features_hs, temp=cfg.KD.DISTILL_TEMPERATURE)
        kl_loss = distill.kl_loss()
        loss_dict['distill_kl'] = kl_loss
        loss_dict['loss'] += cfg.KD.LOSS_WEIGHTS['kl'] * loss_dict['distill_kl']

    # KL loss for classifier
    if 'KL_noavg' in distill_type:
        loss_dict['distill_kl'] = 0
        distill = distillation_loss('KL_noavg', features_ht, features_hs, temp=cfg.KD.DISTILL_TEMPERATURE)
        kl_loss = distill.kl_loss_noavg()
        loss_dict['distill_kl'] = kl_loss
        loss_dict['loss'] += cfg.KD.LOSS_WEIGHTS['kl'] * loss_dict['distill_kl']

    # SSIM loss for heatmaps
    if 'SSIM' in distill_type:
        loss_dict['distill_ssim'] = 0
        distill = distillation_loss('SSIM', features_ht, features_hs)
        ssim_loss = distill.ssim_loss()
        loss_dict['distill_ssim'] = ssim_loss
        loss_dict['loss'] += cfg.KD.LOSS_WEIGHTS['ssim'] * loss_dict['distill_ssim']
    # # Adaptive loss for classifier
    # if 'Adaptive' in distill_type:
    #     distill = distillation_loss('Adaptive', features_ht, features_hs, temp=cfg.MODEL.DISTILL.TEMPERATURE)
    #     adap_loss = distill.adap_loss()
    #     loss_dict['distill_adap'] = adap_loss
    #     loss = loss_dict['loss'] + adap_wt * loss_dict['distill_adap']
    #
    # if 'Adaptive2' in distill_type:
    #     distill = distillation_loss('Adaptive2', features_ht, features_hs, temp=cfg.MODEL.DISTILL.TEMPERATURE)
    #     adap_loss = distill.adap_loss2()
    #     loss_dict['distill_adap2'] = adap_loss
    #     loss = loss_dict['loss'] + adap2_wt * loss_dict['distill_adap2']
    #
    # if 'CE_ST' in distill_type:
    #     distill = distillation_loss('CE_ST', features_ht, features_hs, temp=cfg.MODEL.DISTILL.TEMPERATURE)
    #     ce_loss = distill.ce_student_teacher_loss()
    #     loss_dict['distill_ce'] = ce_loss
    #     loss = loss_dict['loss'] + cest_wt * loss_dict['distill_ce']

    # L2 loss for heatmaps
    if 'L2' in distill_type:
        loss_dict['distill_l2'] = 0
        distill = distillation_loss('L2', features_ht, features_hs)
        l2_loss = distill.l2_loss()
        loss_dict['distill_l2'] = l2_loss
        loss_dict['loss'] += cfg.KD.LOSS_WEIGHTS['l2'] * loss_dict['distill_l2']

    if 'L2_B' in distill_type:
        loss_dict['distill_l2_b'] = 0
        distill = distillation_loss('L2_B', features_bt, features_bs)
        l2_loss = distill.l2_b_loss()
        loss_dict['distill_l2_b'] = l2_loss
        loss_dict['loss'] += cfg.KD.LOSS_WEIGHTS['l2_b'] * loss_dict['distill_l2_b']
        #to-do-put weighted loss for other loss functions also to view on tensorboard
        loss_dict['distill_l2_b'] = cfg.KD.LOSS_WEIGHTS['l2_b'] * loss_dict['distill_l2_b']

    if 'L1_MASK' in distill_type:
        loss_dict['distill_l1_mask'] = 0
        distill = distillation_loss('L2', features_ht, features_hs, targets['boxes'])
        l2_loss = distill.l1_loss_mask()
        loss_dict['distill_l1_mask'] = l2_loss
        loss_dict['loss'] += cfg.KD.LOSS_WEIGHTS['l1'] * loss_dict['distill_l1_mask']

    # L2 loss for center
    if 'L1_REG_MASK' in distill_type:
        loss_dict['distill_l1_reg_mask'] = 0
        distill = distillation_loss('L2', features_ht, features_hs, targets['boxes'])
        l1_loss_reg = distill.l1_loss_reg_mask()
        loss_dict['distill_l1_reg_mask'] = l1_loss_reg
        loss_dict['loss'] += cfg.KD.LOSS_WEIGHTS['l1_reg_mask'] * loss_dict['distill_l1_reg_mask']

    if 'L2_MASK' in distill_type:
        loss_dict['distill_l2_mask'] = 0
        distill = distillation_loss('L2', features_ht, features_hs, targets['boxes'])
        l1_loss = distill.l2_loss_mask()
        loss_dict['distill_l2_mask'] = l1_loss
        loss_dict['loss'] += cfg.KD.LOSS_WEIGHTS['l2_mask'] * loss_dict['distill_l2_mask']

    # L2 loss for center
    if 'L2_REG_MASK' in distill_type:
        loss_dict['distill_l2_reg_mask'] = 0
        distill = distillation_loss('L2', features_ht, features_hs, targets['boxes'])
        l2_loss_reg = distill.l2_loss_reg_mask()
        loss_dict['distill_l2_reg_mask'] = l2_loss_reg
        loss_dict['loss'] += cfg.KD.LOSS_WEIGHTS['l2_reg_mask'] * loss_dict['distill_l2_reg_mask']

    # L2 loss for center
    if 'L2_CENTR_MASK' in distill_type:
        loss_dict['distill_l2_centr_mask'] = 0
        distill = distillation_loss('L2', features_ht, features_hs, targets['boxes'])
        l2_loss_centr = distill.l2_loss_centr_mask()
        loss_dict['distill_l2_centr_mask'] = l2_loss_centr
        loss_dict['loss'] += cfg.KD.LOSS_WEIGHTS['l2_centr_mask'] * loss_dict['distill_l2_centr_mask']

    # L1 loss for heatmaps
    if 'L1' in distill_type:
        loss_dict['distill_l1'] = 0
        distill = distillation_loss('L1', features_ht, features_hs)
        l1_loss = distill.l1_loss()
        loss_dict['distill_l1'] = l1_loss
        loss_dict['loss'] += cfg.KD.LOSS_WEIGHTS['l1_mask'] * loss_dict['distill_l1']

    if 'Sim_Preserve' in distill_type:
        loss_dict['sim_preserve'] = 0
        distill = distillation_loss('sim_preserve', features_ht, features_hs)
        sim_loss = distill.similarity_preserving_loss()
        loss_dict['sim_preserve'] = sim_loss
        loss_dict['loss'] += cfg.KD.LOSS_WEIGHTS['sim_preserve'] * loss_dict['sim_preserve']


class distillation_loss():
    def __init__(self, mode, ft_t, ft_s, target=None, temp=1, at_layers=1):
        self.mode = mode
        self.ft_t = ft_t
        self.ft_s = ft_s
        self.target = target
        self.temp = temp
        self.at_layers = at_layers

    def loss(self):
        if self.mode == 'L2':
            loss = self.l2_loss()
        elif self.mode == 'AT' :
            loss = self.at_loss()
        elif self.mode == 'KL':
            loss = self.kl_loss()
        elif self.mode == 'KL_noavg':
            loss = self.kl_loss_noavg()
        elif self.mode == 'SSIM':
            loss = self.ssim_loss()
        elif self.mode == 'Adaptive':
            loss = self.adap_loss()
        return loss

    def l1_loss(self):
        loss = 0
        # self.ft_t['hm'] = _sigmoid(self.ft_t['hm'])
        # self.ft_s['hm'] = _sigmoid(self.ft_s['hm'])

        hmap_t = self.ft_t['hm']
        hmap_s = self.ft_s['hm']

        criterion_L1 = nn.L1Loss()
        loss = criterion_L1(hmap_s, hmap_t)
        return loss

    def l2_loss(self):
        loss = 0
        # self.ft_t['hm'] = _sigmoid(self.ft_t['hm'])
        # with torch.no_grad():
        #     self.ft_s['hm'] = _sigmoid(self.ft_s['hm'])

        hmap_t = self.ft_t #['hm']
        hmap_s = self.ft_s #['hm']
        loss = criterion_MSE(hmap_s, hmap_t)

        return loss

    def l2_b_loss(self):
        loss = 0

        for layer in range(len(self.ft_s)):
            if self.ft_s[layer].shape[2] != self.ft_t[layer].shape[2] :
                diff = self.ft_t[layer].shape[2] - self.ft_s[layer].shape[2]
                if diff > 0:
                    result = F.pad(self.ft_s[layer], [diff//2, diff//2, diff//2, diff//2])
                    loss += criterion_MSE(result, self.ft_t[layer])
                else:
                    diff = abs(diff)
                    result = F.pad(self.ft_t[layer], pad=(diff//2, diff//2, diff//2, diff//2))
                    loss += criterion_MSE(self.ft_s[layer], result)
            else:
                loss += criterion_MSE(self.ft_s[layer], self.ft_t[layer])
        return loss

    def l1_loss_mask(self):
        loss = 0
        ind = self.target['ind']
        mask = self.target['reg_mask']

        # self.ft_t['hm'] = _sigmoid(self.ft_t['hm'])
        # self.ft_s['hm'] = _sigmoid(self.ft_s['hm'])
        hmap_t = self.ft_t['hm']
        hmap_s = self.ft_s['hm']

        pred = _tranpose_and_gather_feat(hmap_s, ind)
        target = _tranpose_and_gather_feat(hmap_t, ind)
        mask = mask.unsqueeze(2).expand_as(pred).float()
        loss = F.l1_loss(pred * mask, target * mask, size_average=False)
        #loss = criterion_MSE(pred * mask, target * mask)
        loss = loss / (mask.sum() + 1e-4)
        return loss

    def l2_loss_mask(self):
        loss = 0
        ind = self.target['ind']
        mask = self.target['reg_mask']

        # self.ft_t['hm'] = _sigmoid(self.ft_t['hm'])
        # self.ft_s['hm'] = _sigmoid(self.ft_s['hm'])
        hmap_t = self.ft_t['hm']
        hmap_s = self.ft_s['hm']

        pred = _tranpose_and_gather_feat(hmap_s, ind)
        target = _tranpose_and_gather_feat(hmap_t, ind)
        mask = mask.unsqueeze(2).expand_as(pred).float()
        loss = criterion_MSE(pred * mask, target * mask)
        loss = loss / (mask.sum() + 1e-4)
        return loss

    def l1_loss_reg_mask(self):
        loss = 0
        ind = self.target['ind']
        mask = self.target['reg_mask']
        hmap_t = self.ft_t['wh']
        hmap_s = self.ft_s['wh']

        pred = _tranpose_and_gather_feat(hmap_s, ind)
        target = _tranpose_and_gather_feat(hmap_t, ind)

        mask = mask.unsqueeze(2).expand_as(pred).float()

        loss = F.l1_loss(pred * mask, target * mask, size_average=True)
        loss = loss / (mask.sum() + 1e-4)

        #loss = criterion_MSE(hmap_s, hmap_t)

        return loss

    def l2_loss_reg_mask(self):
        loss = 0
        ind = self.target['ind']
        mask = self.target['reg_mask']
        hmap_t = self.ft_t['wh']
        hmap_s = self.ft_s['wh']

        pred = _tranpose_and_gather_feat(hmap_s, ind)
        target = _tranpose_and_gather_feat(hmap_t, ind)

        mask = mask.unsqueeze(2).expand_as(pred).float()

        loss = criterion_MSE(pred * mask, target * mask)
        loss = loss / (mask.sum() + 1e-4)

        return loss

    def l2_loss_centr_mask(self):
        loss = 0
        ind = self.target['ind']
        mask = self.target['reg_mask']
        hmap_t = self.ft_t['reg']
        hmap_s = self.ft_s['reg']

        pred = _tranpose_and_gather_feat(hmap_s, ind)
        target = _tranpose_and_gather_feat(hmap_t, ind)

        mask = mask.unsqueeze(2).expand_as(pred).float()

        loss = criterion_MSE(pred * mask, target * mask)
        loss = loss / (mask.sum() + 1e-4)

        return loss

    def at_loss(self):
        loss = 0

        for layer in range(len(self.ft_s)):
            if self.ft_s[layer].shape[2] != self.ft_t[layer].shape[2] :
                diff = self.ft_t[layer].shape[2] - self.ft_s[layer].shape[2]
                if diff > 0:
                    result = F.pad(self.ft_s[layer], [diff//2, diff//2, diff//2, diff//2])
                    loss += (at(result) - at(self.ft_t[layer])).pow(2).sum()
                else:
                    diff = abs(diff)
                    result = F.pad(self.ft_t[layer], pad=(diff//2, diff//2, diff//2, diff//2))
                    loss += (at(self.ft_s[layer]) - at(result)).pow(2).sum()
            else:
                loss += (at(self.ft_s[layer]) - at(self.ft_t[layer])).pow(2).sum()
        return loss

    def kl_loss(self):
        T = self.temp
        loss = 0
        s = self.ft_s #['hm']
        t = self.ft_t #['hm']
        p = F.log_softmax(s / T, dim=2)  # Change it to 1 for CenterNet
        q = F.softmax(t / T, dim=2)  # Change it to 1 for CenterNet

        loss = F.kl_div(p, q, size_average=False) * (T ** 2)
        loss = loss / (s.shape[0] * s.shape[1])
        return loss

    def kl_loss_noavg(self):
        T = self.temp
        loss = 0
        s = self.ft_s #['hm']
        t = self.ft_t #['hm']
        p = F.log_softmax(s / T, dim=2)  # Change it to 1 for CenterNet
        q = F.softmax(t / T, dim=2)  # Change it to 1 for CenterNet

        loss = F.kl_div(p, q, size_average=False) * (T ** 2) #/ (s.shape[0] * s.shape[1])
        return loss

    def similarity_preserving_loss(self):

        A_t = s = self.ft_t['hm']
        A_s = s = self.ft_s['hm']

        b1, c1, h1, w1 = A_t.shape
        b2, c2, h2, w2 = A_s.shape

        Q_t = A_t.reshape([b1, c1 * h1 * w1])
        Q_s = A_s.reshape([b2, c2 * h2 * w2])

        # evaluate normalized similarity matrices (eq 3)
        G_t = torch.mm(Q_t, Q_t.t())
        # G_t = G_t / G_t.norm(p=2)
        G_t = torch.nn.functional.normalize(G_t)

        G_s = torch.mm(Q_s, Q_s.t())
        # G_s = G_s / G_s.norm(p=2)
        G_s = torch.nn.functional.normalize(G_s)

        # calculate the similarity preserving loss (eq 4)
        loss = (G_t - G_s).pow(2).mean()
        return loss


    # def ssim_loss(self):
    #
    #     hmap_t = self.ft_t['hm']
    #     hmap_s = self.ft_s['hm']
    #     loss = 0
    #
    #     ssim_module = SSIM(data_range=1, size_average=False,
    #                        win_size=11, channel=hmap_t.shape[1])
    #     ssim_loss = 1 - ssim_module(hmap_t, hmap_s)
    #     loss = ssim_loss.mean()
    #
    #     return loss

    def adap_loss2(self):
        kl_loss = self.kl_loss()

        hmap_t = self.ft_t['hm']
        hmap_s = self.ft_s['hm']

        #hmap_t_sig = _sigmoid(hmap_t)
        #hmap_s_sig = _sigmoid(hmap_s)
        #ce_loss = F.cross_entropy(hmap_s, hmap_s_sig.type(torch.LongTensor).cuda(), reduction='sum')

        p = hmap_t_sig = _sigmoid(hmap_t)
        p1 = F.log_softmax(hmap_t)
        pos_loss = torch.log(p) * p
        neg_loss = torch.log(1-p) * (1-p)
        loss = pos_loss + neg_loss
        entropy_loss = 0 - loss.mean()

        # pos_inds = gt.eq(1).float()
        # num_pos = pos_inds.float().sum()
        # loss = 0
        # loss = loss - (pos_loss + neg_loss) / num_pos
        #entropy2 = -np.dot(p, np.log2(p))

        beta = 1.5
        gamma = 2
        dw = kl_loss + beta * entropy_loss
        adw = 1 - torch.exp(-dw)
        adw = adw ** gamma
        adw_loss = adw * kl_loss

        return adw_loss

    def adap_loss(self):
        kl_loss = self.kl_loss()

        gamma = 2
        dw = 1 - torch.exp(-kl_loss)
        dw = dw ** gamma
        dw_loss = dw * kl_loss

        return dw_loss

    def ce_student_teacher_loss(self):

        pred = _sigmoid(self.ft_s['hm'])
        gt = _sigmoid(self.ft_t['hm'])
        neg_weights = torch.pow(1 - gt, 4)

        loss = 0

        pos_loss = torch.log(pred) * torch.pow(1 - pred, 2) * gt
        neg_loss = torch.log(1 - pred) * torch.pow(pred, 2) * neg_weights

        loss = pos_loss + neg_loss
        loss = 0 - loss.mean()

        return loss
