from od.modeling import registry
from .efficient_net import EfficientNet

__all__ = ['efficient_net_b0', 'efficient_net_b1', 'efficient_net_b2', 'efficient_net_b3', 'efficient_net_b4',
           'efficient_net_b5', 'efficient_net_b6', 'efficient_net_b7']


def efficient_net(cfg, pretrained, name):
    if pretrained:
        fn = EfficientNet.from_pretrained
    else:
        fn = EfficientNet.from_name

    return fn(name, enable_extra=cfg.MODEL.BACKBONE.EXTRA)


@registry.BACKBONES.register('efficient_net-b0')
def efficient_net_b0(cfg, pretrained=True):
    """Features from each model :
        Ex. if image size is 512;
        if enable_extra:
            channel : (32,  24, 40, 112, 320, 256, 256, 256)
            output : (256, 128, 64, 32, 16, 8, 6, 4)
        else :
            channel : (32,  24, 40, 112, 320)
            output : (256, 128, 64, 32, 16)
    """
    return efficient_net(cfg, pretrained, 'efficientnet-b0')


@registry.BACKBONES.register('efficient_net-b1')
def efficient_net_b1(cfg, pretrained=True):
    """Features from each model :
        Ex. if image size is 512;
        if enable_extra:
            channel : (32,  24, 40, 112, 320, 256, 256, 256)
            output : (256, 128, 64, 32, 16, 8, 6, 4)
        else :
            channel : (32,  24, 40, 112, 320)
            output : (256, 128, 64, 32, 16)
    """
    return efficient_net(cfg, pretrained, 'efficientnet-b1')


@registry.BACKBONES.register('efficient_net-b2')
def efficient_net_b2(cfg, pretrained=True):
    """Features from each model :
        Ex. if image size is 512;
        if enable_extra:
            channel : (32,  24, 48, 120, 352, 256, 256, 256)
            output : (256, 128, 64, 32, 16, 8, 6, 4)
        else :
            channel : (32,  24, 40, 120, 352)
            output : (256, 128, 64, 32, 16)
    """
    return efficient_net(cfg, pretrained, 'efficientnet-b2')


@registry.BACKBONES.register('efficient_net-b3')
def efficient_net_b3(cfg, pretrained=True):
    """Features from each model :
        Ex. if image size is 512;
        if enable_extra:
            channel : (32,  24, 48, 136, 384, 256, 256, 256)
            output : (256, 128, 64, 32, 16, 8, 6, 4)
        else :
            channel : (32,  24, 40, 136, 384)
            output : (256, 128, 64, 32, 16)
    """
    return efficient_net(cfg, pretrained, 'efficientnet-b3')


@registry.BACKBONES.register('efficient_net-b4')
def efficient_net_b4(cfg, pretrained=True):
    """Features from each model :
        Ex. if image size is 512;
        if enable_extra:
            channel : (48, 32, 56, 160, 448, 256, 256, 256)
            output : (256, 128, 64, 32, 16, 8, 6, 4)
        else :
            channel : (48, 32, 56, 160, 448)
            output : (256, 128, 64, 32, 16)
    """
    return efficient_net(cfg, pretrained, 'efficientnet-b4')


@registry.BACKBONES.register('efficient_net-b5')
def efficient_net_b5(cfg, pretrained=True):
    """Features from each model :
        Ex. if image size is 512;
        if enable_extra:
            channel : (48, 40, 64, 176, 512, 256, 256, 256)
            output : (256, 128, 64, 32, 16, 8, 6, 4)
        else :
            channel : (48, 40, 64, 176, 512)
            output : (256, 128, 64, 32, 16)
    """
    return efficient_net(cfg, pretrained, 'efficientnet-b5')


@registry.BACKBONES.register('efficient_net-b6')
def efficient_net_b6(cfg, pretrained=True):
    return efficient_net(cfg, pretrained, 'efficientnet-b6')


@registry.BACKBONES.register('efficient_net-b7')
def efficient_net_b7(cfg, pretrained=True):
    return efficient_net(cfg, pretrained, 'efficientnet-b7')
