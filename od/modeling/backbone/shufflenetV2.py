import torch
import torch.nn as nn
from od.utils.model_zoo import load_state_dict_from_url
from od.modeling import registry

__all__ = [
    'ShuffleNetV2', 'shufflenet_v2_x0_5', 'shufflenet_v2_x1_0',
    'shufflenet_v2_x1_5', 'shufflenet_v2_x2_0'
]

model_urls = {
    'shufflenetv2_x0.5': 'https://download.pytorch.org/models/shufflenetv2_x0.5-f707e7126e.pth',
    'shufflenetv2_x1.0': 'https://download.pytorch.org/models/shufflenetv2_x1-5666bf0f80.pth',
    'shufflenetv2_x1.5': None,
    'shufflenetv2_x2.0': None,
}


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


class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride):
        super(InvertedResidual, self).__init__()

        if not (1 <= stride <= 3):
            raise ValueError('illegal stride value')
        self.stride = stride

        branch_features = oup // 2
        assert (self.stride != 1) or (inp == branch_features << 1)

        if self.stride > 1:
            self.branch1 = nn.Sequential(
                self.depthwise_conv(inp, inp, kernel_size=3,
                                    stride=self.stride, padding=1),
                nn.BatchNorm2d(inp),
                nn.Conv2d(inp, branch_features, kernel_size=1,
                          stride=1, padding=0, bias=False),
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
            self.depthwise_conv(branch_features, branch_features,
                                kernel_size=3, stride=self.stride, padding=1),
            nn.BatchNorm2d(branch_features),
            nn.Conv2d(branch_features, branch_features,
                      kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(branch_features),
            nn.ReLU(inplace=True),
        )

    @staticmethod
    def depthwise_conv(i, o, kernel_size, stride=1, padding=0, bias=False):
        return nn.Conv2d(i, o, kernel_size, stride, padding, bias=bias, groups=i)

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

class ShuffleNetV2(nn.Module):
    def __init__(self, stages_repeats, stages_out_channels, num_classes=1000, inverted_residual=InvertedResidual, enable_extra=False):
        super(ShuffleNetV2, self).__init__()

        if len(stages_repeats) != 3:
            raise ValueError(
                'expected stages_repeats as list of 3 positive ints')
        if len(stages_out_channels) != 5:
            raise ValueError(
                'expected stages_out_channels as list of 5 positive ints')
        self._stage_out_channels = stages_out_channels

        input_channels = 3
        output_channels = self._stage_out_channels[0]
        self.conv1 = nn.Sequential(
            nn.Conv2d(input_channels, output_channels, 3, 2, 1, bias=False),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True),
        )
        input_channels = output_channels

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        stage_names = ['stage{}'.format(i) for i in [2, 3, 4]]
        for name, repeats, output_channels in zip(
                stage_names, stages_repeats, self._stage_out_channels[1:]):
            seq = [inverted_residual(input_channels, output_channels, 2)]
            for i in range(repeats - 1):
                seq.append(inverted_residual(
                    output_channels, output_channels, 1))
            setattr(self, name, nn.Sequential(*seq))
            input_channels = output_channels

        output_channels = self._stage_out_channels[-1]
        self.conv5 = nn.Sequential(
            nn.Conv2d(input_channels, output_channels, 1, 1, 0, bias=False),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True),
        )

        self.avg = ExtraLayers(output_channels) if enable_extra else nn.AdaptiveAvgPool2d(1) 

    def _forward_impl(self, x):
        feature_maps = list()
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.maxpool(x)
        feature_maps.append(x)
        x = self.stage2(x)
        feature_maps.append(x)
        x = self.stage3(x)
        feature_maps.append(x)
        x = self.stage4(x)
        feature_maps.append(x)
        x = self.conv5(x)
        x = self.avg(x)  # globalpool
        if isinstance(self.avg, nn.AdaptiveAvgPool2d):
            feature_maps.append(x)
        else:
            feature_maps.extend(x)
        return feature_maps

    def forward(self, x):
        return self._forward_impl(x)


def _shufflenetv2(arch, pretrained, *args, **kwargs):
    model = ShuffleNetV2(*args, **kwargs)

    if pretrained:
        model_url = model_urls[arch]
        if model_url is None:
            raise NotImplementedError(
                'pretrained {} is not supported as of now'.format(arch))
        else:
            model.load_state_dict(torch.hub.load_state_dict_from_url(model_url, progress=False),
                                           strict=False)
    return model


@registry.BACKBONES.register('shufflenet_v2_x0_5')
def shufflenet_v2_x0_5(cfg, pretrained=True, **kwargs):
    """
    if enable_extra: 
    output_channels: (24, 48, 96, 192, 512, 512, 512, 512)
    if input size is 512, then the sizes of the outputs are:
    (128, 64, 32, 16, 8, 4, 2, 1)

    else:
    output_channels: (24, 48, 96, 192, 1024)
    if input size is 512, then the sizes of the outputs are:
    (128, 64, 32, 16, 1)
    """
    return _shufflenetv2('shufflenetv2_x0.5', pretrained,
                         [4, 8, 4], [24, 48, 96, 192, 1024], enable_extra=cfg.MODEL.BACKBONE.EXTRA, **kwargs)


@registry.BACKBONES.register('shufflenet_v2_x1_0')
def shufflenet_v2_x1_0(cfg, pretrained=True, **kwargs):
    """
    if enable_extra: 
    output_channels: (24, 116, 232, 464, 512, 512, 512, 512)
    if input size is 512, then the sizes of the outputs are:
    (128, 64, 32, 16, 8, 4, 2, 1)

    else:
    output_channels: (24, 116, 232, 464, 1024)
    if input size is 512, then the sizes of the outputs are:
    (128, 64, 32, 16, 1)
    """
    return _shufflenetv2('shufflenetv2_x1.0', pretrained,
                         [4, 8, 4], [24, 116, 232, 464, 1024], enable_extra=cfg.MODEL.BACKBONE.EXTRA, **kwargs)


@registry.BACKBONES.register('shufflenet_v2_x1_5')
def shufflenet_v2_x1_5(cfg, pretrained=True, **kwargs):
    """
    if enable_extra: 
    output_channels: (24, 176, 352, 704, 512, 512, 512, 512)
    if input size is 512, then the sizes of the outputs are:
    (128, 64, 32, 16, 8, 4, 2, 1)

    else:
    output_channels: (24, 176, 352, 704, 1024)
    if input size is 512, then the sizes of the outputs are:
    (128, 64, 32, 16, 1)
    """
    return _shufflenetv2('shufflenetv2_x1.5', pretrained,
                         [4, 8, 4], [24, 176, 352, 704, 1024], enable_extra=cfg.MODEL.BACKBONE.EXTRA, **kwargs)


@registry.BACKBONES.register('shufflenet_v2_x2_0')
def shufflenet_v2_x2_0(cfg, pretrained=True, **kwargs):
    """
    if enable_extra: 
    output_channels: (24, 244, 488, 976, 512, 512, 512, 512)
    if input size is 512, then the sizes of the outputs are:
    (128, 64, 32, 16, 8, 4, 2, 1)

    else:
    output_channels: (24, 244, 488, 976, 2048)
    if input size is 512, then the sizes of the outputs are:
    (128, 64, 32, 16, 1)
    """
    return _shufflenetv2('shufflenetv2_x2.0', pretrained,
                         [4, 8, 4], [24, 244, 488, 976, 2048], enable_extra=cfg.MODEL.BACKBONE.EXTRA, **kwargs)
