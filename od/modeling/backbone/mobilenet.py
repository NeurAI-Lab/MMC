import os
import torch
from torch import nn
from od.modeling import registry
from collections import OrderedDict
import math

model_urls = {
    'mobilenet_v2': 'https://download.pytorch.org/models/mobilenet_v2-b0353104.pth',
}

model_loc = {
    'mobilenet_v3_small': 'mobilenetv3-small',
    'mobilenet_v3_large': 'mobilenetv3-large',
    'mobilenet_v3_large_0.75' : 'mobilenetv3-large-0.75'
}

class ConvBNReLU(nn.Sequential):
    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1, groups=1):
        padding = (kernel_size - 1) // 2
        super(ConvBNReLU, self).__init__(
            nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding, groups=groups, bias=False),
            nn.BatchNorm2d(out_planes),
            nn.ReLU6(inplace=True)
        )


class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        hidden_dim = int(round(inp * expand_ratio))
        self.use_res_connect = self.stride == 1 and inp == oup

        layers = []
        if expand_ratio != 1:
            # pw
            layers.append(ConvBNReLU(inp, hidden_dim, kernel_size=1))
        layers.extend([
            # dw
            ConvBNReLU(hidden_dim, hidden_dim, stride=stride, groups=hidden_dim),
            # pw-linear
            nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
            nn.BatchNorm2d(oup),
        ])
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class MobileNetV2(nn.Module):
    def __init__(self, width_mult=1.0, enable_extra=False, prune=False, inverted_residual_setting=None):
        super(MobileNetV2, self).__init__()
        block = InvertedResidual
        input_channel = 32
        if prune:
            last_channel = 960
        else:
            last_channel = 1280

        if inverted_residual_setting is None:
            inverted_residual_setting = [
                # t, c, n, s
                [1, 16, 1, 1],
                [6, 24, 2, 2],
                [6, 32, 3, 2],
                [6, 64, 4, 2],
                [6, 96, 3, 1],
                [6, 160, 3, 2],
                [6, 320, 1, 1],
            ]

        # only check the first element, assuming user knows t,c,n,s are required
        if len(inverted_residual_setting) == 0 or len(inverted_residual_setting[0]) != 4:
            raise ValueError("inverted_residual_setting should be non-empty "
                             "or a 4-element list, got {}".format(inverted_residual_setting))

        # building first layer
        input_channel = int(input_channel * width_mult)
        self.last_channel = int(last_channel * max(1.0, width_mult))
        features = [ConvBNReLU(3, input_channel, stride=2)]
        # building inverted residual blocks
        for t, c, n, s in inverted_residual_setting:
            output_channel = int(c * width_mult)
            for i in range(n):
                stride = s if i == 0 else 1
                features.append(block(input_channel, output_channel, stride, expand_ratio=t))
                input_channel = output_channel
        # building last several layers
        features.append(ConvBNReLU(input_channel, self.last_channel, kernel_size=1))
        # make it nn.Sequential
        self.features = nn.Sequential(*features)
        self.extras = None
        if enable_extra:
            self.extras = nn.ModuleList([
                InvertedResidual(self.last_channel, 512, 2, 0.2),
                InvertedResidual(512, 256, 2, 0.25),
                InvertedResidual(256, 256, 2, 0.5),
                InvertedResidual(256, 64, 2, 0.25)
            ])

        self.reset_parameters()

    def reset_parameters(self):
        # weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        features = []
        for i in range(14):
            x = self.features[i](x)
            features.append(x)

        for i in range(14, len(self.features)):
            x = self.features[i](x)
            features.append(x)

        if self.extras:
            for i in range(len(self.extras)):
                x = self.extras[i](x)
                features.append(x)
            return features[1], features[3], features[6], features[13], features[18], features[19], \
                       features[20], features[21], features[22]

        """Features from each model :
        Ex. if image size is 512;
        if enable_extra:
            channel : (16,  24, 32, 96, 1280, 512, 256, 256, 64)
            output : (255, 128, 64, 32, 16, 8, 4, 2, 1)
        else :
            channel : (16,  24, 32, 96, 1280)
            output : (255, 128, 64, 32, 16)
        """
        return features[1], features[3],features[6],features[13],features[18]


def _make_divisible(v, divisor, min_value=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    :param v:
    :param divisor:
    :param min_value:
    :return:
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v

class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6

class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)

class SELayer(nn.Module):
    def __init__(self, channel, reduction=4):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, _make_divisible(channel // reduction, 8)),
            nn.ReLU(inplace=True),
            nn.Linear(_make_divisible(channel // reduction, 8), channel),
            h_sigmoid(),
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y

def conv_3x3_bn(inp, oup, stride):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False), nn.BatchNorm2d(oup), h_swish()
    )

def conv_1x1_bn(inp, oup):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False), nn.BatchNorm2d(oup), h_swish()
    )

class InvertedResidual_v3(nn.Module):
    def __init__(self, inp, hidden_dim, oup, kernel_size, stride, use_se, use_hs):
        super(InvertedResidual_v3, self).__init__()
        assert stride in [1, 2]

        self.identity = stride == 1 and inp == oup

        if inp == hidden_dim:
            self.conv = nn.Sequential(
                # dw
                nn.Conv2d(
                    hidden_dim,
                    hidden_dim,
                    kernel_size,
                    stride,
                    (kernel_size - 1) // 2,
                    groups=hidden_dim,
                    bias=False,
                ),
                nn.BatchNorm2d(hidden_dim),
                h_swish() if use_hs else nn.ReLU(inplace=True),
                # Squeeze-and-Excite
                SELayer(hidden_dim) if use_se else nn.Sequential(),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )
        else:
            self.conv = nn.Sequential(
                # pw
                nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
                nn.BatchNorm2d(hidden_dim),
                h_swish() if use_hs else nn.ReLU(inplace=True),
                # dw
                nn.Conv2d(
                    hidden_dim,
                    hidden_dim,
                    kernel_size,
                    stride,
                    (kernel_size - 1) // 2,
                    groups=hidden_dim,
                    bias=False,
                ),
                nn.BatchNorm2d(hidden_dim),
                # Squeeze-and-Excite
                SELayer(hidden_dim) if use_se else nn.Sequential(),
                h_swish() if use_hs else nn.ReLU(inplace=True),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )

    def forward(self, x):
        if self.identity:
            return x + self.conv(x)
        else:
            return self.conv(x)

class MobileNetV3(nn.Module):
    def __init__(self, cfgs, mode, enable_extra=False, num_classes=1000, width_mult=1.0):
        super(MobileNetV3, self).__init__()
        # setting of inverted residual blocks
        self.cfgs = cfgs
        assert mode in ["large", "small"]

        # building first layer
        input_channel = _make_divisible(16 * width_mult, 8)
        all_layers = []
        layers = [conv_3x3_bn(3, input_channel, 2)]

        # building inverted residual blocks
        block = InvertedResidual_v3
        temp = 1
        for k, exp_size, c, use_se, use_hs, s in self.cfgs:
            if use_se == temp:
                all_layers.append(nn.Sequential(*layers))
                layers = []
                temp = 1 - temp

            output_channel = _make_divisible(c * width_mult, 8)
            layers.append(
                block(input_channel, exp_size, output_channel, k, s, use_se, use_hs)
            )
            input_channel = output_channel

        all_layers.append(nn.Sequential(*layers))

        self.features = nn.Sequential(*all_layers)
        # building last several layers
        self.conv = nn.Sequential(
            conv_1x1_bn(input_channel, _make_divisible(exp_size * width_mult, 8)),
            SELayer(_make_divisible(exp_size * width_mult, 8))
            if mode == "small"
            else nn.Sequential(),
        )

        #self.avgpool = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)), h_swish())
        output_channel = {'large': 960, 'small': 576}
        output_channel = (
            _make_divisible(output_channel[mode] * width_mult, 8) if width_mult > 1.0 else output_channel[mode]
        )

        # self.conv1 = nn.Sequential(
        #     conv_3x3_bn(exp_size, _make_divisible(output_channel * width_mult, 8), 2),
        # )
        if enable_extra:
            self.extras = nn.ModuleList([
                InvertedResidual(output_channel[mode], 512, 2, 0.2),
                InvertedResidual(512, 256, 2, 0.25),
                InvertedResidual(256, 256, 2, 0.5),
                InvertedResidual(256, 64, 2, 0.25),
                InvertedResidual(64, 64, 2, 0.25)
            ])

        self._initialize_weights()

    def forward(self, x):
        features = []
        for i in range(len(self.features)):
            x = self.features[i](x)
            features.append(x)

        x = self.conv(x)
        features.append(x)
        #x = self.conv1(x)
        #features.append(x)

        if self.extras:
            for i in range(len(self.extras)):
                x = self.extras[i](x)
                features.append(x)

        return features

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

class MobileNetV3_official(nn.Module):
    def __init__(self, cfgs, mode, enable_extra=False, num_classes=1000, width_mult=1.0):
        super(MobileNetV3_official, self).__init__()
        # setting of inverted residual blocks
        self.cfgs = cfgs
        assert mode in ["large", "small"]

        self.strides_at = []
        # building first layer
        input_channel = _make_divisible(16 * width_mult, 8)
        layers = [conv_3x3_bn(3, input_channel, 2)]

        # building inverted residual blocks
        block = InvertedResidual_v3
        for idx, (k, t, c, use_se, use_hs, s) in enumerate(self.cfgs):
            if s == 2:
                self.strides_at.append(idx + 1)
            output_channel = _make_divisible(c * width_mult, 8)
            exp_size = _make_divisible(input_channel * t, 8)
            layers.append(
                block(input_channel, exp_size, output_channel, k, s, use_se, use_hs)
            )
            input_channel = output_channel

        self.features = nn.Sequential(*layers)
        self.conv = conv_1x1_bn(input_channel, exp_size)

        output_channel = {'large': 960, 'small': 576}
        output_channel = _make_divisible(output_channel[mode] * width_mult,8) if width_mult < 1.0 else output_channel[mode]
        if enable_extra:
            self.extras = nn.ModuleList([
                InvertedResidual(output_channel, 512, 2, 0.2),
                InvertedResidual(512, 256, 2, 0.25),
                InvertedResidual(256, 64, 2, 0.25),
                InvertedResidual(64, 64, 2, 0.25)
            ])
        self._initialize_weights()

    def forward(self, x):
        features = []
        for ft in self.features:
            x = ft(x)
            features.append(x)

        x = self.conv(x)
        features.append(x)

        if self.extras:
            for i in range(len(self.extras)):
                x = self.extras[i](x)
                features.append(x)

        return features

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

@registry.BACKBONES.register('mobilenet_v3_large')
def mobilenetv3_large(cfg, pretrained=True, **kwargs):
    """
    Constructs a MobileNetV3-Large model
    """
    cfgs = [
        # k, t, c, SE, NL, s
        [3, 16, 16, 0, 0, 1],
        [3, 64, 24, 0, 0, 2],
        [3, 72, 24, 0, 0, 1],
        [5, 72, 40, 1, 0, 2],
        [5, 120, 40, 1, 0, 1],
        [5, 120, 40, 1, 0, 1],
        [3, 240, 80, 0, 1, 2],
        [3, 200, 80, 0, 1, 1],
        [3, 184, 80, 0, 1, 1],
        [3, 184, 80, 0, 1, 1],
        [3, 480, 112, 1, 1, 1],
        [3, 672, 112, 1, 1, 1],
        [5, 672, 160, 1, 1, 1],
        [5, 672, 160, 1, 1, 1],  # NOTE Input is set to 7^2 x 112
        [5, 960, 160, 1, 1, 1],
    ]
    model = MobileNetV3(cfgs, mode="large", enable_extra=cfg.MODEL.BACKBONE.EXTRA, **kwargs)
    if pretrained:
        weights_path = os.path.join(cfg.MODEL.BACKBONE.WEIGHTS_PATH, "%s.pth" % model_loc['mobilenet_v3_large'])
        state_dict = torch.load(weights_path)
        model.load_state_dict(state_dict, strict=False)
        print("Pre-trained weights loaded successfully.")

    return model

@registry.BACKBONES.register('mobilenet_v3_small')
def mobilenetv3_small(cfg, pretrained=True, **kwargs):
    """
    Constructs a MobileNetV3-Small model
    """
    cfgs = [
        # k, t, c, SE, NL, s
        [3, 16, 16, 1, 0, 2],  # NOTE Input is set to 112^2 x 24
        [3, 72, 24, 0, 0, 2],  # NOTE Input is set to 112^2 x 24
        [3, 88, 24, 0, 0, 1],
        [5, 96, 40, 1, 1, 2],  # NOTE stride is set to 1
        [5, 240, 40, 1, 1, 1],
        [5, 240, 40, 1, 1, 1],
        [5, 120, 48, 1, 1, 1],
        [5, 144, 48, 1, 1, 1],
        [5, 288, 96, 1, 1, 1],
        [5, 576, 96, 1, 1, 1],
        [5, 576, 96, 1, 1, 1],
    ]

    model = MobileNetV3(cfgs, mode="small", enable_extra=cfg.MODEL.BACKBONE.EXTRA, **kwargs)
    if pretrained:
        weights_path = os.path.join(cfg.MODEL.BACKBONE.WEIGHTS_PATH, "%s.pth" % model_loc['mobilenet_v3_small'])
        state_dict = torch.load(weights_path)
        model.load_state_dict(state_dict, strict=False)
        print("Pre-trained weights loaded successfully.")

    return model

@registry.BACKBONES.register('mobilenet_v3_large_official')
def mobilenetv3_large_official(cfg, pretrained=True, **kwargs):
    """
    Constructs a MobileNetV3-Large model
    """
    cfgs = [
        # k, t, c, SE, HS, s
        [3, 1, 16, 0, 0, 1],
        [3, 4, 24, 0, 0, 2],
        [3, 3, 24, 0, 0, 1],
        [5, 3, 40, 1, 0, 2],
        [5, 3, 40, 1, 0, 1],
        [5, 3, 40, 1, 0, 1],
        [3, 6, 80, 0, 1, 2],
        [3, 2.5, 80, 0, 1, 1],
        [3, 2.3, 80, 0, 1, 1],
        [3, 2.3, 80, 0, 1, 1],
        [3, 6, 112, 1, 1, 1],
        [3, 6, 112, 1, 1, 1],
        [5, 6, 160, 1, 1, 2],
        [5, 6, 160, 1, 1, 1],
        [5, 6, 160, 1, 1, 1]
    ]
    model = MobileNetV3_official(cfgs, mode="large", enable_extra=cfg.MODEL.BACKBONE.EXTRA, width_mult=cfg.MODEL.BACKBONE.WIDTH_MULT, **kwargs)
    if pretrained:
        weights = OrderedDict()
        weights_path = os.path.join(cfg.MODEL.BACKBONE.WEIGHTS_PATH, "%s.pth" % model_loc['mobilenet_v3_large'])
        model_name = 'mobilenet_v3_large' if cfg.MODEL.BACKBONE.WIDTH_MULT==1 else 'mobilenet_v3_large_0.75'
        weights_path = os.path.join(cfg.MODEL.BACKBONE.WEIGHTS_PATH, "%s.pth" % model_loc[model_name])
        loaded = torch.load(weights_path)
        for i, v in loaded.items():
            if 'classifier' in i:
                continue
            weights[i] = v

        model.load_state_dict(weights, strict=False)
    return model

@registry.BACKBONES.register('mobilenet_v3_small_official')
def mobilenetv3_small_official(cfg, pretrained=True, **kwargs):
    """
    Constructs a MobileNetV3-Large model
    """
    cfgs = [
        # k, t, c, SE, HS, s
        [3, 1, 16, 1, 0, 2],
        [3, 4.5, 24, 0, 0, 2],
        [3, 3.67, 24, 0, 0, 1],
        [5, 4, 40, 1, 1, 2],
        [5, 6, 40, 1, 1, 1],
        [5, 6, 40, 1, 1, 1],
        [5, 3, 48, 1, 1, 1],
        [5, 3, 48, 1, 1, 1],
        [5, 6, 96, 1, 1, 2],
        [5, 6, 96, 1, 1, 1],
        [5, 6, 96, 1, 1, 1],
    ]
    model = MobileNetV3_official(cfgs, mode="small", enable_extra=cfg.MODEL.BACKBONE.EXTRA, width_mult=cfg.MODEL.BACKBONE.WIDTH_MULT, **kwargs)
    if pretrained:
        weights = OrderedDict()
        weights_path = os.path.join(cfg.MODEL.BACKBONE.WEIGHTS_PATH, "%s.pth" % model_loc['mobilenet_v3_small'])
        loaded = torch.load(weights_path)
        for i, v in loaded.items():
            if 'classifier' in i:
                continue
            weights[i] = v

        model.load_state_dict(weights, strict=False)
    return model

@registry.BACKBONES.register('mobilenet_v2')
def mobilenet_v2(cfg, pretrained=True, **kwargs):
    """Features from each model :
    Ex. if image size is 512;
    if enable_extra:
        channel : (16,  24, 32, 96, 1280, 512, 256, 256, 64)
        output : (255, 128, 64, 32, 16, 8, 4, 2, 1)
    else :
        channel : (16,  24, 32, 96, 1280)
        output : (255, 128, 64, 32, 16)
    """
    width_mult = cfg.MODEL.BACKBONE.WIDTH_MULT
    model = MobileNetV2(width_mult = width_mult, enable_extra=cfg.MODEL.BACKBONE.EXTRA, **kwargs)
    if pretrained:
        model.load_state_dict(torch.hub.load_state_dict_from_url(model_urls['mobilenet_v2'], progress=False),
                              strict=False)
        print("Pre-trained weights loaded successfully.")

    return model

@registry.BACKBONES.register('mobilenet_v2_prune')
def mobilenet_v2_prune(cfg, pretrained=True, **kwargs):
    """Features from each model :
    Ex. if image size is 512;
    if enable_extra:
        channel : (16,  24, 32, 96, 1280, 512, 256, 256, 64)
        output : (255, 128, 64, 32, 16, 8, 4, 2, 1)
    else :
        channel : (16,  24, 32, 96, 1280)
        output : (255, 128, 64, 32, 16)
    """
    width_mult = cfg.MODEL.BACKBONE.WIDTH_MULT
    model = MobileNetV2(width_mult = width_mult, enable_extra=cfg.MODEL.BACKBONE.EXTRA, prune=True, **kwargs)
    if pretrained:
        model_dict = model.state_dict()
        pretrained_dict = torch.hub.load_state_dict_from_url(model_urls['mobilenet_v2'], progress=False)
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict
                           and (model_dict[k].shape == pretrained_dict[k].shape)}

        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)

        #model.load_state_dict(torch.hub.load_state_dict_from_url(model_urls['mobilenet_v2'], progress=False),strict=False)
        print("Pre-trained weights loaded successfully.")

    return model
