import os
import torch
import torch.nn as nn
from od.modeling import registry

__all__ = ['SNet', 'snet49', 'snet146', 'snet535']

CONFIG = {
    49: {"conv1_inp": 24, "outp_per_stage": [60, 120, 240], "stages_repeats": [4, 8, 4]}, # [3, 7, 3]
    146: {"conv1_inp": 24, "outp_per_stage": [132, 264, 528], "stages_repeats": [4, 8, 4]}, # [3, 7, 3]
    535: {"conv1_inp": 48, "outp_per_stage": [248, 496, 992], "stages_repeats": [4, 8, 4]}, # [3, 7, 3]
    } 

PRE_TRAINED = {
    49:  "SNet49",
    146: "SNet146",
    535: "SNet535",
    }
##########
# To get all pre-trained weights, please visit https://confluence.navinfo.eu/display/NIE/SNet+Pretrained
# How to load the pretrained weights:
# On local machine: specify the path to the pre-trained weights in PRE_TRAINED
# On kubectl: assuming you mount "/data/output/dummy" to "/output" on the server and have include the pre-trained weights in this directory,
#             then your path will be "/output/$(name_of_the_weights)" and follow the instruction for local machines
##########


class SNet(nn.Module):
    def __init__(self, model_size: int, num_classes: int = 1000, mode: str = "backbone", enable_extra=False):
        super(SNet, self).__init__()
        self.mode = mode
        if model_size in CONFIG:
            config = CONFIG[model_size]
        else:
            raise ValueError("Model of specified size NOT implemented!")

        inp = 3
        self.conv1 = ConvBNReLu(inp, config["conv1_inp"], kernel_size=3, stride=2, padding=1)
        self.max_pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        inp = config["conv1_inp"]

        stage_names = ['stage{}'.format(i) for i in [2, 3, 4]]
        for name, repeats, outp in zip(
                stage_names, config["stages_repeats"], config["outp_per_stage"]):
            seq = [InvertedResidual(inp, outp, kernel_size=5, stride=2, padding=2)]
            for i in range(repeats - 1):
                seq.append(InvertedResidual(outp, outp, kernel_size=5, stride=1, padding=2))
            setattr(self, name, nn.Sequential(*seq))
            inp = outp

        if model_size == 49:
            outp = 512
            self.conv5 = ConvBNReLu(inp, outp, kernel_size=1, stride=1, padding=0)
            inp = outp
        else:
            self.conv5 = None

        self.extra_layers = None
        self.avg_pool = None
        self.fc = None
        if self.mode == "cls":
            self.avg_pool = nn.AdaptiveAvgPool2d(1)
            self.fc = nn.Linear(inp, num_classes)
        else:
            if enable_extra:
                self.extra_layers = ExtraLayers(inp)
            else:
                self.avg_pool = nn.AdaptiveAvgPool2d(1)
            self.fc = None

    def forward(self, x):
        if self.mode != "cls":
            feature_maps = list()
            x = self.conv1(x)
            feature_maps.append(x)
            x = self.max_pool(x)
            feature_maps.append(x)
            x = self.stage2(x)
            feature_maps.append(x)
            x = self.stage3(x)
            feature_maps.append(x)
            x = self.stage4(x)
            if self.conv5 is None:
                feature_maps.append(x)
            else:
                x = self.conv5(x)
                feature_maps.append(x)

            if self.extra_layers is not None:
                extra_features = self.extra_layers(x)
                feature_maps.extend(extra_features)
            else:
                x = self.avg_pool(x)
                feature_maps.append(x)

            return feature_maps
        else:
            x = self.conv1(x)
            x = self.max_pool(x)
            x = self.stage2(x)
            x = self.stage3(x)
            x = self.stage4(x)
            if self.conv5 is not None:
                x = self.conv5(x)
            x = self.avg_pool(x)
            x = self.fc(x.view(x.size(0), -1))
            return x

    def load_state_dict(self, state_dict):  # soft loading for state_dict
        super(SNet, self).load_state_dict(state_dict, strict=False)


class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, kernel_size, stride, padding):
        super(InvertedResidual, self).__init__()

        if not (1 <= stride <= 3):
            raise ValueError('illegal stride value')
        self.stride = stride

        branch_features = oup // 2
        assert (self.stride != 1) or (inp == branch_features << 1)

        if self.stride > 1:
            self.branch1 = nn.Sequential(
                nn.Conv2d(inp, inp, kernel_size=kernel_size, stride=self.stride, padding=padding, bias=False,
                          groups=inp),
                nn.BatchNorm2d(inp),
                nn.Conv2d(inp, branch_features, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(branch_features),
                nn.ReLU(inplace=True),
            )
        else:
            self.branch1 = nn.Sequential()

        self.branch2 = nn.Sequential(
            nn.Conv2d(inp if (self.stride > 1) else branch_features,
                      branch_features, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(branch_features),
            nn.ReLU(inplace=True),
            nn.Conv2d(branch_features, branch_features, kernel_size=kernel_size, stride=self.stride, padding=padding,
                      bias=False,
                      groups=branch_features),
            nn.BatchNorm2d(branch_features),
            nn.Conv2d(branch_features, branch_features, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(branch_features),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        if self.stride == 1:
            x1, x2 = x.chunk(2, dim=1)
            out = torch.cat((x1, self.branch2(x2)), dim=1)
        else:
            out = torch.cat((self.branch1(x), self.branch2(x)), dim=1)

        out = channel_shuffle(out, 2)

        return out


class ExtraLayers(nn.Module):
    def __init__(self, inp):
        super(ExtraLayers, self).__init__()
        self.convbnrelu1_1 = ConvBNReLu(inp, 256, kernel_size=1, stride=1, padding=0)
        self.convbnrelu1_2 = ConvBNReLu(256, 512, kernel_size=3, stride=2, padding=1)
        self.convbnrelu2_1 = ConvBNReLu(512, 256, kernel_size=1, stride=1, padding=0)
        self.convbnrelu2_2 = ConvBNReLu(256, 512, kernel_size=3, stride=2, padding=1)
        self.convbnrelu3_1 = ConvBNReLu(512, 256, kernel_size=1, stride=1, padding=0)
        self.convbnrelu3_2 = ConvBNReLu(256, 512, kernel_size=3, stride=2, padding=1)
        self.avgpool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        feature_maps = list()
        x = self.convbnrelu1_1(x)
        x = self.convbnrelu1_2(x)
        feature_maps.append(x)
        x = self.convbnrelu2_1(x)
        x = self.convbnrelu2_2(x)
        feature_maps.append(x)
        x = self.convbnrelu3_1(x)
        x = self.convbnrelu3_2(x)
        feature_maps.append(x)
        x = self.avgpool(x)
        feature_maps.append(x)
        return feature_maps


class ConvBNReLu(nn.Module):
    def __init__(self, inp, outp, kernel_size, stride, padding):
        super(ConvBNReLu, self).__init__()
        self.op = nn.Sequential(
            nn.Conv2d(inp, outp, kernel_size=kernel_size, stride=stride, padding=padding, bias=False),
            nn.BatchNorm2d(outp),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.op(x)


@registry.BACKBONES.register('snet49')
def snet49(cfg, pre_trained=False, num_classes: int = None, mode: str = "backbone"):
    """
    if enable_extra: 
    output_channels: (24, 24, 60, 120, 512, 512, 512, 512, 512)
    if input size is 512, then the sizes of the outputs are:
    (256, 128, 64, 32, 16, 8, 4, 2, 1)

    else:
    output_channels: (24, 24, 60, 120, 512, 512)
    if input size is 512, then the sizes of the outputs are:
    (256, 128, 64, 32, 16, 1)
    """
    model = SNet(model_size=49, num_classes=num_classes, mode=mode, enable_extra=cfg.MODEL.BACKBONE.EXTRA)
    if pre_trained:
        weights_path = os.path.join(cfg.MODEL.BACKBONE.WEIGHTS_PATH, "%s.tar" % PRE_TRAINED[49])
        saved = torch.load(weights_path)

        model.load_state_dict(saved["state_dict"])
        print("Pre-trained weights loaded successfully.")

    return model


@registry.BACKBONES.register('snet146')
def snet146(cfg, pre_trained=False, num_classes: int = None, mode: str = "backbone"):
    """
    if enable_extra: 
    output_channels: (24, 24, 132, 264, 528, 512, 512, 512, 512)
    if input size is 512, then the sizes of the outputs are:
    (256, 128, 64, 32, 16, 8, 4, 2, 1)

    else:
    output_channels: (24, 24, 132, 264, 528, 528)
    if input size is 512, then the sizes of the outputs are:
    (256, 128, 64, 32, 16, 1)
    """
    model = SNet(model_size=146, num_classes=num_classes, mode=mode, enable_extra=cfg.MODEL.BACKBONE.EXTRA)
    if pre_trained:
        weights_path = os.path.join(cfg.MODEL.BACKBONE.WEIGHTS_PATH, "%s.tar" % PRE_TRAINED[146])
        saved = torch.load(weights_path)

        model.load_state_dict(saved["state_dict"])
        print("Pre-trained weights loaded successfully.")

    return model


@registry.BACKBONES.register('snet535')
def snet535(cfg, pre_trained=False, num_classes: int = None, mode: str = "backbone"):
    """
    if enable_extra: 
    output_channels: (48, 48, 248, 496, 992, 512, 512, 512, 512)
    if input size is 512, then the sizes of the outputs are:
    (256, 128, 64, 32, 16, 8, 4, 2, 1)

    else:
    output_channels: (48, 48, 248, 496, 992, 992)
    if input size is 512, then the sizes of the outputs are:
    (256, 128, 64, 32, 16, 1)
    """
    model = SNet(model_size=535, num_classes=num_classes, mode=mode, enable_extra=cfg.MODEL.BACKBONE.EXTRA)
    if pre_trained:
        weights_path = os.path.join(cfg.MODEL.BACKBONE.WEIGHTS_PATH, "%s.tar" % PRE_TRAINED[535])
        saved = torch.load(weights_path)

        model.load_state_dict(saved["state_dict"])
        print("Pre-trained weights loaded successfully.")

    return model


def channel_shuffle(x, groups):
    # type: (torch.Tensor, int) -> torch.Tensor
    batchsize, num_channels, height, width = x.data.size()
    channels_per_group = num_channels // groups

    # reshape
    x = x.view(batchsize, groups,
               channels_per_group, height, width)

    x = torch.transpose(x, 1, 2).contiguous()

    # flatten
    x = x.view(batchsize, -1, height, width)

    return x
