import torch.nn as nn
import torch.nn.functional as F
import torch

from od.layers import L2Norm
from od.modeling import registry
#from torchsummary import summary

class conv_bn_relu(nn.Module):
    def __init__(self, nin, nout, kernel_size, stride, padding, bias=False):
        super(conv_bn_relu, self).__init__()
        self.conv = nn.Conv2d(nin, nout, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)
        self.batch_norm = nn.BatchNorm2d(nout)
        self.relu = nn.ReLU(True)

    def forward(self, x):
        out = self.conv(x)
        out = self.batch_norm(out)
        out = self.relu(out)

        return out

class StemBlock(nn.Module):
    def __init__(self, init_feature=32):
        super(StemBlock, self).__init__()

        self.conv_3x3_first = conv_bn_relu(nin=3, nout=init_feature, kernel_size=3, stride=2, padding=1, bias=False)

        self.conv_1x1_left = conv_bn_relu(nin=init_feature, nout=int(init_feature/2), kernel_size=1, stride=1, padding=0, bias=False)
        self.conv_3x3_left = conv_bn_relu(nin=int(init_feature/2), nout=init_feature, kernel_size=3, stride=2, padding=1, bias=False)

        self.max_pool_right = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        self.conv_1x1_last = conv_bn_relu(nin=2*(init_feature), nout=init_feature, kernel_size=1, stride=1, padding=0, bias=False)

    def forward(self, x):
        out_first = self.conv_3x3_first(x)

        out_left = self.conv_1x1_left(out_first)
        out_left = self.conv_3x3_left(out_left)

        out_right = self.max_pool_right(out_first)

        out_middle = torch.cat((out_left, out_right), 1)

        out_last = self.conv_1x1_last(out_middle)

        return out_last


class Transition_layer(nn.Sequential):
    def __init__(self, nin, theta=1):
        super(Transition_layer, self).__init__()

        self.add_module('conv_1x1',
                        conv_bn_relu(nin=nin, nout=int(nin * theta), kernel_size=1, stride=1, padding=0, bias=False))
        #self.add_module('avg_pool_2x2', nn.AvgPool2d(kernel_size=2, stride=2, padding=0))


class dense_layer(nn.Module):
    def __init__(self, nin, growth_rate, bottleneck_width , drop_rate=0.05):
        super(dense_layer, self).__init__()

        growth_rate = growth_rate // 2

        self.inter_channel = int(growth_rate * bottleneck_width/4) * 4

        if self.inter_channel > nin/2:
            self.inter_channel = int(nin/8) * 4

        self.dense_left_way = nn.Sequential()

        self.dense_left_way.add_module('conv_1x1',
                                       conv_bn_relu(nin=nin, nout=self.inter_channel, kernel_size=1, stride=1, padding=0,
                                                    bias=False))
        self.dense_left_way.add_module('conv_3x3',
                                       conv_bn_relu(nin=self.inter_channel, nout=growth_rate, kernel_size=3, stride=1,
                                                    padding=1, bias=False))

        self.dense_right_way = nn.Sequential()

        self.dense_right_way.add_module('conv_1x1',
                                        conv_bn_relu(nin=nin, nout=self.inter_channel, kernel_size=1, stride=1, padding=0,
                                                     bias=False))
        self.dense_right_way.add_module('conv_3x3_1',
                                        conv_bn_relu(nin=self.inter_channel, nout=growth_rate, kernel_size=3,
                                                     stride=1, padding=1, bias=False))
        self.dense_right_way.add_module('conv_3x3 2',
                                        conv_bn_relu(nin=growth_rate, nout=growth_rate, kernel_size=3,
                                                     stride=1, padding=1, bias=False))

        self.drop_rate = drop_rate

    def forward(self, x):
        left_output = self.dense_left_way(x)
        right_output = self.dense_right_way(x)

        if self.drop_rate > 0:
            left_output = F.dropout(left_output, p=self.drop_rate, training=self.training)
            right_output = F.dropout(right_output, p=self.drop_rate, training=self.training)

        dense_layer_output = torch.cat((x, left_output, right_output), 1)

        return dense_layer_output


class DenseBlock(nn.Sequential):
    def __init__(self, nin, num_dense_layers, growth_rate, bottleneck_width = 4, drop_rate=0.0):
        super(DenseBlock, self).__init__()

        for i in range(num_dense_layers):
            nin_dense_layer = nin + growth_rate * i
            self.add_module('dense_layer_%d' % i,
                            dense_layer(nin=nin_dense_layer, growth_rate=growth_rate, bottleneck_width=bottleneck_width ,drop_rate=drop_rate))


class Subsampling(nn.Sequential):
    def __init__(self, nin, growth_rate,pad = 1, stride = 2, drop_rate=0.0):
        super(Subsampling, self).__init__()

        self.add_module('conv1', conv_bn_relu(nin=nin, nout=int( growth_rate* 16), kernel_size=1, stride=1, padding=0, bias=False))
        self.add_module('conv2', conv_bn_relu(nin=growth_rate* 16, nout=int( growth_rate* 16), kernel_size=3, stride=stride, padding=pad, bias=False))

class ResNetBlock_bottleneck(nn.Module):
    def __init__( self, in_planes, planes=256, stride=1):
        super(ResNetBlock_bottleneck, self).__init__()
        self.left_conv1 = nn.Conv2d(in_channels = in_planes, out_channels = int(planes/2), kernel_size= 1, stride = 1)
        self.left_bn1 = nn.BatchNorm2d(int(planes/2))
        self.left_conv2 =  nn.Conv2d(in_channels = int(planes/2), out_channels = int(planes/2), kernel_size=3, padding=1)
        self.left_bn2 = nn.BatchNorm2d(int(planes/2))
        self.left_conv3 =  nn.Conv2d(in_channels = int(planes/2), out_channels = planes, kernel_size=1)
        self.left_bn3 = nn.BatchNorm2d(planes)

        self.right_conv1 = nn.Conv2d(in_channels = int(in_planes), out_channels = int(planes), kernel_size=1)
        self.right_bn1 = nn.BatchNorm2d(planes)

        self.conv = nn.Conv2d(in_channels = planes, out_channels = planes, kernel_size= 1, stride = 1)

    def forward(self, x):
        left = F.relu(self.left_bn1(self.left_conv1(x)))
        left = F.relu(self.left_bn2(self.left_conv2(left)))
        left = F.relu(self.left_bn3(self.left_conv3(left)))

        right = F.relu(self.right_bn1(self.right_conv1(x)))

        out = left + right
        out = (self.conv(out))
        return out

class PeleeNet(nn.Module):
    def __init__(self, growth_rate=32, num_dense_layers=[3, 4, 8, 6], init_feat= 32 ,theta=1, drop_rate=0.0, enable_extra=False):
        super(PeleeNet, self).__init__()

        self.bottleneck_width=[1,2,4,4]
        self.init_feat = init_feat

        self.trace = []

        assert len(num_dense_layers) == 4

        # start an ModuleList for PeleeNet backbone
        self.peleenet = nn.ModuleList()
        self.ResBlocks = nn.ModuleList()

        self.peleenet.append(StemBlock(init_feature= self.init_feat))

        # we shall append the 2 Res blocks for the feature output from the first 2 stages
        self.ResBlocks.append(ResNetBlock_bottleneck(in_planes=128))
        self.ResBlocks.append(ResNetBlock_bottleneck(in_planes=256))

        nin_transition_layer = self.init_feat

        for i in range(len(num_dense_layers)):
            block_def = nn.Sequential()
            block_def.add_module('DenseBlock_%d' % (i),
                                     DenseBlock(nin=nin_transition_layer, num_dense_layers=num_dense_layers[i],
                                                growth_rate=growth_rate, bottleneck_width=self.bottleneck_width[i], drop_rate=0.0))
            nin_transition_layer += num_dense_layers[i] * growth_rate

            if i == len(num_dense_layers) - 1:
                block_def.add_module('Transition_layer_%d' % (i),
                                         conv_bn_relu(nin=nin_transition_layer, nout=int(nin_transition_layer * theta),
                                                      kernel_size=1, stride=1, padding=0, bias=False))
            else:
                block_def.add_module('Transition_layer_%d' % (i),
                                         Transition_layer(nin=nin_transition_layer, theta=1))

            self.peleenet.append(block_def)

            if i > 1:
                self.ResBlocks.append(ResNetBlock_bottleneck(in_planes=nin_transition_layer))
                if i == 2:
                    self.ResBlocks.append(ResNetBlock_bottleneck(in_planes=nin_transition_layer))

            if (i != len(num_dense_layers) - 1):
                self.peleenet.append(nn.AvgPool2d(kernel_size=2, stride=2))

        if enable_extra:
            self.peleenet.append(Subsampling(nin=nin_transition_layer, growth_rate=growth_rate, pad = 1, stride = 2))
            nin_subsampling = growth_rate * 16
            self.ResBlocks.append(ResNetBlock_bottleneck(in_planes=nin_subsampling))

            self.peleenet.append(Subsampling(nin=nin_subsampling, growth_rate=growth_rate, pad = 0, stride = 1))
            self.ResBlocks.append(ResNetBlock_bottleneck(in_planes=nin_subsampling))

            self.peleenet.append(Subsampling(nin=nin_subsampling, growth_rate=growth_rate, pad = 0, stride = 1))
            self.ResBlocks.append(ResNetBlock_bottleneck(in_planes=nin_subsampling))

    def forward(self, x):
        features = []
        feature_loc = [1, 3, 5, 7, 8, 9, 10] #[4,6,8,9,10,11]
        layer_count = 0
        ResNet_block_count = 0
        for layer in self.peleenet:
            x = layer(x)
            if layer_count in feature_loc:
                ResBlock = self.ResBlocks[ResNet_block_count]
                features.append(ResBlock(x))
                ResNet_block_count += 1
                if layer_count == 5:
                    features.append(ResBlock(x))
                    ResNet_block_count += 1

            layer_count += 1
        return tuple(features)

@registry.BACKBONES.register('peleenet')
def peleenet(cfg, pretrained=False):
    """Features from each model :
        Ex. if image size is 512;
        if enable_extra:
            channel : (256, 256, 256, 256, 256, 256, 256, 256)
            output : (128, 64, 32, 32, 16, 8, 6, 4)
        else :
            channel : (256, 256, 256, 256, 256)
            output : (128, 64, 32, 32, 16)
    """
    model = PeleeNet(enable_extra=cfg.MODEL.BACKBONE.EXTRA)
    return model

