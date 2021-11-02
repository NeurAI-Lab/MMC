import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from od.modeling import registry

__all__ = ['HarDNet', 'HarDNet39', 'HarDNet68', 'HarDNet85']

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

class Flatten(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x.view(x.data.size(0), -1)


class CombConvLayer(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel=1, stride=1, dropout=0.1, bias=False):
        super().__init__()
        self.add_module('layer1', ConvLayer(in_channels, out_channels, kernel))
        self.add_module('layer2', DWConvLayer(
            out_channels, out_channels, stride=stride))

    def forward(self, x):
        return super().forward(x)


class DWConvLayer(nn.Sequential):
    def __init__(self, in_channels, out_channels,  stride=1,  bias=False):
        super().__init__()
        out_ch = out_channels

        groups = in_channels
        kernel = 3
        #print(kernel, 'x', kernel, 'x', out_channels, 'x', out_channels, 'DepthWise')

        self.add_module('dwconv', nn.Conv2d(groups, groups, kernel_size=3,
                                            stride=stride, padding=1, groups=groups, bias=bias))
        self.add_module('norm', nn.BatchNorm2d(groups))

    def forward(self, x):
        return super().forward(x)


class ConvLayer(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel=3, stride=1, dropout=0.1, bias=False):
        super().__init__()
        out_ch = out_channels
        groups = 1
        self.kernel = kernel
        #print(kernel, 'x', kernel, 'x', in_channels, 'x', out_channels)
        self.add_module('conv', nn.Conv2d(in_channels, out_ch, kernel_size=kernel,
                                          stride=stride, padding=kernel//2, groups=groups, bias=bias))
        self.add_module('norm', nn.BatchNorm2d(out_ch))
        self.add_module('relu', nn.ReLU6(True))

    def forward(self, x):
        return super().forward(x)


class HarDBlock(nn.Module):
    def get_link(self, layer, base_ch, growth_rate, grmul):
        if layer == 0:
            return base_ch, 0, []
        out_channels = growth_rate
        link = []
        for i in range(10):
            dv = 2 ** i
            if layer % dv == 0:
                k = layer - dv
                link.append(k)
                if i > 0:
                    out_channels *= grmul
        out_channels = int(int(out_channels + 1) / 2) * 2
        in_channels = 0
        for i in link:
            ch, _, _ = self.get_link(i, base_ch, growth_rate, grmul)
            in_channels += ch
        return out_channels, in_channels, link

    def get_out_ch(self):
        return self.out_channels

    def __init__(self, in_channels, growth_rate, grmul, n_layers, keepBase=False, residual_out=False, dwconv=False):
        super().__init__()
        self.keepBase = keepBase
        self.links = []
        layers_ = []
        self.out_channels = 0  # if upsample else in_channels
        for i in range(n_layers):
            outch, inch, link = self.get_link(
                i+1, in_channels, growth_rate, grmul)
            self.links.append(link)
            use_relu = residual_out
            if dwconv:
                layers_.append(CombConvLayer(inch, outch))
            else:
                layers_.append(ConvLayer(inch, outch))

            if (i % 2 == 0) or (i == n_layers - 1):
                self.out_channels += outch
        #print("Blk out =",self.out_channels)
        self.layers = nn.ModuleList(layers_)

    def forward(self, x):
        layers_ = [x]

        for layer in range(len(self.layers)):
            link = self.links[layer]
            tin = []
            for i in link:
                tin.append(layers_[i])
            if len(tin) > 1:
                x = torch.cat(tin, 1)
            else:
                x = tin[0]
            out = self.layers[layer](x)
            layers_.append(out)

        t = len(layers_)
        out_ = []
        for i in range(t):
            if (i == 0 and self.keepBase) or \
               (i == t-1) or (i % 2 == 1):
                out_.append(layers_[i])
        out = torch.cat(out_, 1)
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

class HarDNet(nn.Module):
    def __init__(self, depth_wise=False, arch=85, pretrained=True, weight_path='', enable_extra=False):
        super().__init__()
        first_ch = [32, 64]
        second_kernel = 3
        max_pool = True
        grmul = 1.7
        drop_rate = 0.1

        # HarDNet68
        ch_list = [128, 256, 320, 640, 1024]
        gr = [14, 16, 20, 40, 160]
        n_layers = [8, 16, 16, 16,  4]
        downSamp = [1,  0,  1,  1,  0]

        if arch == 85:
            # HarDNet85
            first_ch = [48, 96]
            ch_list = [192, 256, 320, 480, 720, 1280]
            gr = [24,  24,  28,  36,  48, 256]
            n_layers = [8,  16,  16,  16,  16,   4]
            downSamp = [1,   0,   1,   0,   1,   0]
            drop_rate = 0.2
        elif arch == 39:
            # HarDNet39
            first_ch = [24, 48]
            ch_list = [96, 320, 640, 1024]
            grmul = 1.6
            gr = [16,  20, 64, 160]
            n_layers = [4,  16,  8,   4]
            downSamp = [1,   1,  1,   0]

        if depth_wise:
            second_kernel = 1
            max_pool = False
            drop_rate = 0.05

        blks = len(n_layers)
        self.base = nn.ModuleList([])

        # First Layer: Standard Conv3x3, Stride=2
        self.base.append(
            ConvLayer(in_channels=3, out_channels=first_ch[0], kernel=3,
                      stride=2,  bias=False))

        # Second Layer
        self.base.append(
            ConvLayer(first_ch[0], first_ch[1],  kernel=second_kernel))

        # Maxpooling or DWConv3x3 downsampling
        if max_pool:
            self.base.append(nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
        else:
            self.base.append(DWConvLayer(first_ch[1], first_ch[1], stride=2))

        # Build all HarDNet blocks
        ch = first_ch[1]
        for i in range(blks):
            blk = HarDBlock(ch, gr[i], grmul, n_layers[i], dwconv=depth_wise)
            ch = blk.get_out_ch()
            self.base.append(blk)

            if i == blks-1 and arch == 85:
                self.base.append(nn.Dropout(0.1))

            self.base.append(ConvLayer(ch, ch_list[i], kernel=1))
            ch = ch_list[i]
            if downSamp[i] == 1:
                if max_pool:
                    self.base.append(nn.MaxPool2d(kernel_size=2, stride=2))
                else:
                    self.base.append(DWConvLayer(ch, ch, stride=2))

        ch = ch_list[blks-1]
        self.base.append(ExtraLayers(ch) if enable_extra else nn.AdaptiveAvgPool2d((1, 1))) 

        if pretrained:
            if hasattr(torch, 'hub'):

                if arch == 68 and not depth_wise:
                    checkpoint = 'https://ping-chao.com/hardnet/hardnet68-5d684880.pth'
                elif arch == 85 and not depth_wise:
                    checkpoint = 'https://ping-chao.com/hardnet/hardnet85-a28faa00.pth'
                elif arch == 68 and depth_wise:
                    checkpoint = 'https://ping-chao.com/hardnet/hardnet68ds-632474d2.pth'
                else:
                    checkpoint = 'https://ping-chao.com/hardnet/hardnet39ds-0e6c6fa9.pth'

                self.load_state_dict(torch.hub.load_state_dict_from_url(
                    checkpoint, progress=False), strict=False)
            else:
                postfix = 'ds' if depth_wise else ''
                weight_file = '%shardnet%d%s.pth' % (
                    weight_path, arch, postfix)
                if not os.path.isfile(weight_file):
                    print(weight_file, 'is not found')
                    exit(0)
                weights = torch.load(weight_file)
                self.load_state_dict(weights)

            postfix = 'DS' if depth_wise else ''
            print('ImageNet pretrained weights for HarDNet%d%s is loaded' %
                  (arch, postfix))

    def forward(self, x):
        outputs = []
        old_size = 9999999999999
        for layer in self.base:
            x = layer(x)
            if isinstance(layer, ConvLayer):
                if layer.kernel == 1:
                    if x.size(-1) == old_size:
                        outputs.pop()
                    outputs.append(x)
                    old_size = x.size(-1)
            elif isinstance(layer, nn.AdaptiveAvgPool2d):
                outputs.append(x)
            elif isinstance(layer, ExtraLayers):
                outputs.extend(x)
        return outputs


@registry.BACKBONES.register('HarDNet39')
def HarDNet39(cfg, pretrained=False, **kwargs):
    """
    if enable_extra: 
    output_channels: (96, 320, 640, 1024, 512, 512, 512, 512)
    if input size is 512, then the sizes of the outputs are:
    (128, 64, 32, 16, 8, 4, 2, 1)

    else:
    output_channels: (96, 320, 640, 1024, 1024)
    if input size is 512, then the sizes of the outputs are:
    (128, 64, 32, 16, 1)
    """
    model = HarDNet(arch=39, pretrained=pretrained, depth_wise=True, enable_extra=cfg.MODEL.BACKBONE.EXTRA, **kwargs)

    return model


@registry.BACKBONES.register('HarDNet68')
def HarDNet68(cfg, pretrained=False, **kwargs):
    """
    if enable_extra: 
    output_channels: (128, 320, 640, 1024, 512, 512, 512, 512)
    if input size is 512, then the sizes of the outputs are:
    (128, 64, 32, 16, 8, 4, 2, 1)

    else:
    output_channels: (128, 320, 640, 1024, 1024)
    if input size is 512, then the sizes of the outputs are:
    (128, 64, 32, 16, 1)
    """
    model = HarDNet(arch=68, pretrained=pretrained, enable_extra=cfg.MODEL.BACKBONE.EXTRA, **kwargs)

    return model


@registry.BACKBONES.register('HarDNet85')
def HarDNet85(cfg, pretrained=False, **kwargs):
    """
    if enable_extra: 
    output_channels: (192, 320, 720, 1280, 512, 512, 512, 512)
    if input size is 512, then the sizes of the outputs are:
    (128, 64, 32, 16, 8, 4, 2, 1)

    else:
    output_channels: (192, 320, 720, 1280, 1280)
    if input size is 512, then the sizes of the outputs are:
    (128, 64, 32, 16, 1)
    """
    model = HarDNet(arch=85, pretrained=pretrained, enable_extra=cfg.MODEL.BACKBONE.EXTRA, **kwargs)

    return model
