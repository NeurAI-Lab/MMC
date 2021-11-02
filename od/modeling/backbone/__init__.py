from od.modeling import registry
from .vgg import VGG

from .mobilenet import MobileNetV2, MobileNetV3, MobileNetV3_official
from .efficient_net import EfficientNet
from .resnet import ResNet
from .resnext import ResNext
from .xception import Xception
from .peleenet import PeleeNet
from .vovnet import VoVNet
from .darknet import DarkNet19
from .darknet.darknet53 import DarkNet53
from .snet import SNet
from .shufflenetV2 import ShuffleNetV2
from .hardnet import HarDNet
#from .pvtransformer import PyramidVisionTransformer
from .transformers.transformers import DistilledVisionTransformer,VisionTransformer

__all__ = ['build_backbone', 'VGG', 'MobileNetV2', 'MobileNetV3', 'MobileNetV3_official', 'EfficientNet', 'ResNet', 'ResNext', 'Xception', 'PeleeNet',
           'DarkNet19', 'DarkNet53', 'SNet', 'VoVNet', 'ShuffleNetV2', 'HarDNet','DistilledVisionTransformer','VisionTransformer'] #, 'PyramidVisionTransformer']

def build_backbone(cfg):
    return registry.BACKBONES[cfg.MODEL.BACKBONE.NAME](cfg, cfg.MODEL.BACKBONE.PRETRAINED)
