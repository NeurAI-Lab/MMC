import torch
from torch import nn

from od.modeling.backbone import build_backbone
from od.modeling.head import build_head
from od.modeling.detector.build_customized_func import build_customized_func, concat_reduction, concat_nin

from od.modeling.detector.choose_features import choose_features, choose_features_backbone
from od.modeling.detector.reconstruction import Reconstruction

class TwoPhasesDetector(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        # Build backbone
        self.backbone = build_backbone(cfg)

        # Calc backbone param no
        if cfg.LOGGER.DEBUG_MODE:
            import numpy as np
            model_parameters = filter(lambda p: p.requires_grad, self.backbone.parameters())
            params = sum([np.prod(p.size()) for p in model_parameters])
            print('Backbone params:', params)

        # Freeze backbone
        if cfg.MODEL.BACKBONE.FREEZE:
            for param in self.backbone.parameters():
                param.requires_grad = False

        # Build head
        self.features_backbone_kd = choose_features_backbone(cfg)
        self.head = build_head(cfg)

        # Calc head param no
        if cfg.LOGGER.DEBUG_MODE:
            model_parameters = filter(lambda p: p.requires_grad, self.head.parameters())
            params = sum([np.prod(p.size()) for p in model_parameters])
            print('Head params:', params)

        self.features_for_head = choose_features(cfg)
        if self.cfg.KD.CONCAT_FEATURES:
             self.concat_reduction = concat_nin(cfg)
        #self.concat_reduction = concat_reduction(cfg)

        self.customized_func = build_customized_func(cfg)
        if self.cfg.KD.AUX_RECON:
            self.recon = Reconstruction(cfg)

    def forward(self, images, targets=None, teacher=False, modifiedfts=None):

        if not self.training and self.cfg.KD.CONCAT_INPUT:
            images = torch.cat((images, images), 1)

        features_orig = self.backbone(images)
        if self.training:
            if self.cfg.KD.SWAP_FEATURES:
                if modifiedfts is not None:
                    features_orig = modifiedfts
                else:
                    return features_orig
            if self.cfg.KD.CONCAT_FEATURES:
                if modifiedfts is not None:
                    features = modifiedfts
                else:
                    return self.features_for_head(features_orig)

        if not self.training:
            if self.cfg.KD.RECON_PLOT:
                recon = Reconstruction(self.cfg)
                features_backbone = self.features_for_head(features_orig)
                recon_img, _ = recon(features_backbone, None)

                # features_backbone = self.features_for_head(features_orig)
                # features_decode = self.decoder(features_backbone)
                # recon_img = self.decPredictor(features_decode)
                # recon_img = back_transform(recon_img, self.cfg, scale=255)
                b = torch.squeeze(recon_img)
                b = b.permute(1, 2, 0)
                c = b.cpu().detach().numpy()
                from PIL import Image
                pil_img = Image.fromarray(c, "RGB")
                pil_img.save('dl/recon.jpg')

        if not self.cfg.KD.CONCAT_FEATURES:
            features = self.features_for_head(features_orig)
        if not self.training and self.cfg.KD.CONCAT_FEATURES:
            features = self.features_for_head(features_orig)

        if self.training:
            if self.cfg.KD.CONCAT_FEATURES:
                features = self.concat_reduction(features)
        kwargs = self.customized_func(images)
        detections = self.head(features, targets, teacher, **kwargs)

        if self.training:
            features_head = detections
            if self.cfg.KD.ENABLE:
                features_backbone = self.features_backbone_kd(features_orig)
                if self.training:
                    if self.cfg.KD.AUX_RECON:
                        return features_backbone, features_head, features
                    return features_backbone, features_head
            else:
                return detections

        return detections[0]
