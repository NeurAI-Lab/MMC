import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
from torch.nn import  BatchNorm2d as BatchNorm2d
#from torch.nn import BatchNorm2d
from od.modeling import registry

__all__ = ['VoVNet', 'vovnet27_slim', 'vovnet39', 'vovnet57']
# model_urls = {}

model_urls = {
    'vovnet39': 'vovnet39_torchvision',
    'vovnet57': 'vovnet57_torchvision'
}

class ConvBnReluLayer(nn.Module):

    def __init__(self, inplanes, planes, kernel_size, padding, stride, bias=False):
        super(ConvBnReluLayer, self).__init__()
        self.conv = nn.Conv2d(inplanes, planes, kernel_size=kernel_size, padding=padding,
                              stride=stride, bias=bias)
        self.bn = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        out = self.relu(out)
        return out


class ExtraLayers(nn.Module):

    def __init__(self, inplanes):
        super(ExtraLayers, self).__init__()
        self.convbnrelu1_1 = ConvBnReluLayer(inplanes, 256, kernel_size=1, padding=0, stride=1)
        self.convbnrelu1_2 = ConvBnReluLayer(256, 512, kernel_size=3, padding=1, stride=2)
        self.convbnrelu2_1 = ConvBnReluLayer(512, 256, kernel_size=1, padding=0, stride=1)
        self.convbnrelu2_2 = ConvBnReluLayer(256, 512, kernel_size=3, padding=1, stride=2)
        self.convbnrelu3_1 = ConvBnReluLayer(512, 256, kernel_size=1, padding=0, stride=1)
        self.convbnrelu3_2 = ConvBnReluLayer(256, 512, kernel_size=3, padding=1, stride=2)
        self.avgpool = nn.AvgPool2d(2, stride=2)

    def forward(self, x):
        out1_1 = self.convbnrelu1_1(x)
        out1_2 = self.convbnrelu1_2(out1_1)
        out2_1 = self.convbnrelu2_1(out1_2)
        out2_2 = self.convbnrelu2_2(out2_1)
        out3_1 = self.convbnrelu3_1(out2_2)
        out3_2 = self.convbnrelu3_2(out3_1)
        out = self.avgpool(out3_2)
        return [out1_2, out2_2, out3_2, out]



def conv3x3(in_channels, out_channels, module_name, postfix,
            stride=1, groups=1, kernel_size=3, padding=1):
    """3x3 convolution with padding"""
    return [
        ('{}_{}/conv'.format(module_name, postfix),
            nn.Conv2d(in_channels, out_channels,
                      kernel_size=kernel_size,
                      stride=stride,
                      padding=padding,
                      groups=groups,
                      bias=False)),
        ('{}_{}/norm'.format(module_name, postfix),
            BatchNorm2d(out_channels)),
        ('{}_{}/relu'.format(module_name, postfix),
            nn.ReLU(inplace=True)),
    ]


def conv1x1(in_channels, out_channels, module_name, postfix,
            stride=1, groups=1, kernel_size=1, padding=0):
    """1x1 convolution"""
    return [
        ('{}_{}/conv'.format(module_name, postfix),
            nn.Conv2d(in_channels, out_channels,
                      kernel_size=kernel_size,
                      stride=stride,
                      padding=padding,
                      groups=groups,
                      bias=False)),
        ('{}_{}/norm'.format(module_name, postfix),
            BatchNorm2d(out_channels)),
        ('{}_{}/relu'.format(module_name, postfix),
            nn.ReLU(inplace=True)),
    ]


class _OSA_module(nn.Module):
    def __init__(self,
                 in_ch,
                 stage_ch,
                 concat_ch,
                 layer_per_block,
                 module_name,
                 identity=False):
        super(_OSA_module, self).__init__()

        self.identity = identity
        self.layers = nn.ModuleList()
        in_channel = in_ch
        for i in range(layer_per_block):
            self.layers.append(nn.Sequential(
                OrderedDict(conv3x3(in_channel, stage_ch, module_name, i))))
            in_channel = stage_ch

        # feature aggregation
        in_channel = in_ch + layer_per_block * stage_ch
        self.concat = nn.Sequential(
            OrderedDict(conv1x1(in_channel, concat_ch, module_name, 'concat')))

    def forward(self, x):
        identity_feat = x
        output = []
        output.append(x)
        for layer in self.layers:
            x = layer(x)
            output.append(x)

        x = torch.cat(output, dim=1)
        xt = self.concat(x)

        if self.identity:
            xt = xt + identity_feat

        return xt


class _OSA_stage(nn.Sequential):
    def __init__(self,
                 in_ch,
                 stage_ch,
                 concat_ch,
                 block_per_stage,
                 layer_per_block,
                 stage_num):
        super(_OSA_stage, self).__init__()

        if not stage_num == 2:
            self.add_module('Pooling',
                nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True))

        module_name = f'OSA{stage_num}_1'
        self.add_module(module_name,
            _OSA_module(in_ch,
                        stage_ch,
                        concat_ch,
                        layer_per_block,
                        module_name))
        for i in range(block_per_stage-1):
            module_name = f'OSA{stage_num}_{i+2}'
            self.add_module(module_name,
                _OSA_module(concat_ch,
                            stage_ch,
                            concat_ch,
                            layer_per_block,
                            module_name,
                            identity=True))


class VoVNet(nn.Module):
    def __init__(self,
                 config_stage_ch,
                 config_concat_ch,
                 block_per_stage,
                 layer_per_block,
                 num_classes=1000, enable_extra=False):
        super(VoVNet, self).__init__()

        # Stem module
        stem = conv3x3(3,   64, 'stem', '1', 2)
        stem += conv3x3(64,  64, 'stem', '2', 1)
        stem += conv3x3(64, 128, 'stem', '3', 2)
        self.add_module('stem', nn.Sequential(OrderedDict(stem)))
        self.extra_layers =(ExtraLayers(config_concat_ch[-1]) if enable_extra else nn.AdaptiveAvgPool2d((1, 1)))

        stem_out_ch = [128]
        in_ch_list = stem_out_ch + config_concat_ch[:-1]
        self.stage_names = []
        for i in range(4): #num_stages
            name = 'stage%d' % (i+2)
            self.stage_names.append(name)
            self.add_module(name,
                            _OSA_stage(in_ch_list[i],
                                       config_stage_ch[i],
                                       config_concat_ch[i],
                                       block_per_stage[i],
                                       layer_per_block,
                                       i+2))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, (BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.stem(x)
        outs = []
        for name in self.stage_names:
            x = getattr(self, name)(x)
            outs.append(x)
        if isinstance(self.extra_layers, nn.AdaptiveAvgPool2d):
            outs.append(self.extra_layers(x))

        elif isinstance(self.extra_layers, ExtraLayers):
            outs.extend(self.extra_layers(x))

        return outs


def _vovnet(cfg,
            arch,
            config_stage_ch,
            config_concat_ch,
            block_per_stage,
            layer_per_block,
            pretrained,
            progress,enable_extra=False,
            **kwargs):
    model = VoVNet(config_stage_ch, config_concat_ch,
                   block_per_stage, layer_per_block,enable_extra=enable_extra,
                   **kwargs)
    if pretrained and arch in model_urls:
        weights_path = os.path.join(cfg.MODEL.BACKBONE.WEIGHTS_PATH, "%s.pth" % model_urls[arch])
        state_dict = torch.load(weights_path)
        new_state_dict = OrderedDict()
        for key, value in state_dict.items():
            new_state_dict[key[7:]] = value

        model.load_state_dict(new_state_dict, strict=False)
    return model

@registry.BACKBONES.register('vovnet57')
def vovnet57(cfg,pretrained=False, progress=True, **kwargs):
    """
    if enable_extra:
    output_channels: (256, 512, 768, 1024, 512, 512, 512, 512)
    if input size is 512, then the sizes of the outputs are:
    (128, 64, 32, 16, 8, 4, 2, 1)

    else:
    output_channels: (256, 512, 768, 1024, 1024)
    if input size is 512, then the sizes of the outputs are:
    (128, 64, 32, 16, 1)
    """
    model = _vovnet(cfg, 'vovnet57', [128, 160, 192, 224], [256, 512, 768, 1024],
                    [1,1,4,3], 5, pretrained, progress, enable_extra=cfg.MODEL.BACKBONE.EXTRA, **kwargs)

    model.base_inchannels = [256, 512, 768, 1024]
    return model

@registry.BACKBONES.register('vovnet39')
def vovnet39(cfg,pretrained=False, progress=True, **kwargs):
    """
    if enable_extra:
    output_channels: (256, 512, 768, 1024, 512, 512, 512, 512)
    if input size is 512, then the sizes of the outputs are:
    (128, 64, 32, 16, 8, 4, 2, 1)

    else:
    output_channels: (256, 512, 768, 1024, 1024)
    if input size is 512, then the sizes of the outputs are:
    (128, 64, 32, 16, 1)
    """
    model = _vovnet(cfg, 'vovnet39', [128, 160, 192, 224], [256, 512, 768, 1024],
                    [1,1,2,2], 5, pretrained, progress, enable_extra=cfg.MODEL.BACKBONE.EXTRA, **kwargs)
    model.base_inchannels = [256, 512, 768, 1024]
    return model

@registry.BACKBONES.register('vovnet27_slim')
def vovnet27_slim(cfg,pretrained=False, progress=True, **kwargs):
    """
    if enable_extra:
    output_channels: (128, 256, 384, 512, 512, 512, 512, 512)
    if input size is 512, then the sizes of the outputs are:
    (128, 64, 32, 16, 8, 4, 2, 1)

    else:
    output_channels: (128, 256, 384, 512, 512)
    if input size is 512, then the sizes of the outputs are:
    (128, 64, 32, 16, 1)
    """
    return _vovnet(cfg, 'vovnet27_slim', [64, 80, 96, 112], [128, 256, 384, 512],
                    [1,1,1,1], 5, pretrained, progress,  enable_extra=cfg.MODEL.BACKBONE.EXTRA,**kwargs)


