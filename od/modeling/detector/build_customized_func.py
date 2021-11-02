import torch
from torch import nn

class build_customized_func(nn.Module):

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

    def forward(self, images):

        if self.cfg.MODEL.HEAD.NAME == "ThunderNetHead":
            def get_im_info(images):
                im_info = torch.Tensor((images.size(2), images.size(3), 0)).to(
                    images.get_device(), dtype=torch.float32)

                if self.cfg.EXPORT == "onnx":
                    im_info = torch.stack([im_info] * int(images.size(0)))
                else:
                    im_info = torch.stack([im_info] * images.size(0))

                return dict(im_info=im_info)

            return get_im_info(images)

        return dict()


class BasicBlock(nn.Module):
    def __init__(self, inplanes, planes, kernel_size=1):
        super(BasicBlock, self).__init__()
        self.conv = nn.Conv2d(inplanes, planes, kernel_size=kernel_size, bias=False)

    def forward(self, x):
        out = self.conv(x)
        return out

# class concat_reduction(nn.Module):
#
#     def __init__(self, cfg):
#         super().__init__()
#         self.cfg = cfg
#
#         layers = []
#         for i, layer in enumerate(self.cfg.KD.CONCAT_LAYERS):
#             inplanes = cfg.MODEL.BACKBONE.OUT_CHANNELS[i] * 2
#             outplanes = cfg.MODEL.BACKBONE.OUT_CHANNELS[i]
#             layers.append(BasicBlock(inplanes, outplanes))
#
#         self.blocks = nn.ModuleList(layers)
#
#     def forward(self, features):
#
#         mod_features = []
#         if self.cfg.KD.CONCAT_FEATURES:
#             for i, block in enumerate(self.blocks):
#                 mod_features.append(block(features[i]))
#             return mod_features
#
#         return features

class concat_reduction(nn.Module):

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        inplanes = cfg.MODEL.BACKBONE.OUT_CHANNELS[-1] * 2
        outplanes = cfg.MODEL.BACKBONE.OUT_CHANNELS[-1]
        self.conv = nn.Conv2d(inplanes, outplanes, kernel_size=1, bias=False)

    def forward(self, features):

        if self.cfg.KD.CONCAT_FEATURES:
            layer = self.cfg.KD.CONCAT_LAYERS[0]
            out = features[layer]
            features[layer] = self.conv(out)
            return features

        return features

def nin_block( in_channels, out_channels, kernel_size=1, padding=0):
    return nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size, bias=False, padding=padding),
        nn.ReLU(),
        nn.Conv2d(out_channels, out_channels, kernel_size=1),
        nn.ReLU(),
        nn.Conv2d(out_channels, out_channels, kernel_size=1),
        nn.ReLU())


class concat_nin(nn.Module):

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        layers = []
        for i, layer in enumerate(self.cfg.KD.CONCAT_LAYERS):
            inplanes = cfg.MODEL.BACKBONE.OUT_CHANNELS[layer] * 2
            outplanes = cfg.MODEL.BACKBONE.OUT_CHANNELS[layer]
            kernel_size = 3
            padding = 1
            if layer == 5:
                kernel_size = 1
                padding = 0
            layers.append(nin_block(inplanes, outplanes, kernel_size=kernel_size, padding=padding))

        self.blocks = nn.ModuleList(layers)

    def forward(self, features):

        if self.cfg.KD.CONCAT_FEATURES:
            for layer, block in zip(self.cfg.KD.CONCAT_LAYERS, self.blocks):
                features[layer] = block(features[layer])
            return features

        return features