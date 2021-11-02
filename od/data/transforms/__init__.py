from od.modeling.anchors.prior_box import PriorBox
from od.data.transforms.transforms import *

"""
First we list out the transforms for each head
1) ThunderNet
2) SSD
3) YOLO v2 & v3
4) CenterNet
5) EfficientDet
"""


def transform_ThunderNet(cfg, is_train=True):
    """
    Data transforms for ThunderNetHead
    :param cfg: config file
    :param is_train: parameter to differentiate between train and test mode
    :return: transformed data
    """
    if is_train:
        transform = [
            ConvertFromInts(),
            PhotometricDistort(),
            Expand(cfg.INPUT.PIXEL_MEAN),
            RandomSampleCrop(),
            RandomMirror(),
            ResizeImageBoxes(cfg, True),
            SubtractMeans(cfg.INPUT.PIXEL_MEAN),
            ToTensor(),
        ]
    else:
        transform = [
            ResizeImageBoxes(cfg, False),
            SubtractMeans(cfg.INPUT.PIXEL_MEAN),
            ToTensor(),
        ]
    transform = Compose(transform)
    return transform


def transform_SSD(cfg, is_train=True):
    """
    Data transforms for SSD box head
    :param cfg: config file
    :param is_train: parameter to differentiate between train and test mode
    :return: transformed data
    """
    if is_train:
        transform = [
            ConvertFromInts(),
            PhotometricDistort(),
            Expand(cfg.INPUT.PIXEL_MEAN),
            RandomSampleCrop(),
            RandomMirror(),
            ToPercentCoords(),
            Resize(cfg.INPUT.IMAGE_SIZE),
            #SubtractMeans(cfg.INPUT.PIXEL_MEAN),
            Standardize(cfg.INPUT.STANDARDIZATION_MEAN, cfg.INPUT.STANDARDIZATION_STDEV), #only for transformer
            ToTensor(),
        ]
    else:
        transform = [
            Resize(cfg.INPUT.IMAGE_SIZE),
            #SubtractMeans(cfg.INPUT.PIXEL_MEAN),
            Standardize(cfg.INPUT.STANDARDIZATION_MEAN, cfg.INPUT.STANDARDIZATION_STDEV), #only for transformer
            ToTensor()
        ]
    transform = Compose(transform)
    return transform


def transform_FCOS(cfg, is_train=True):
    """
    Data transforms for SSD box head
    :param cfg: config file
    :param is_train: parameter to differentiate between train and test mode
    :return: transformed data
    """
    if is_train:
        if cfg.INPUT.AUTO_AUG=="v1":
            transform = [
                BbAug(cfg.INPUT.AUTO_AUG_VERSION),
                ConvertFromInts(),
                ToPercentCoords(),
                Resize(cfg.INPUT.IMAGE_SIZE),
                Remake(cfg.INPUT.IMAGE_SIZE),
                Standardize(cfg.INPUT.STANDARDIZATION_MEAN, cfg.INPUT.STANDARDIZATION_STDEV),
                ToTensor(),
            ]
        elif cfg.INPUT.AUTO_AUG=="v2":
            transform = [
                BbAug(cfg.INPUT.AUTO_AUG_VERSION),
                ConvertFromInts(),
                PhotometricDistort(),
                Expand(cfg.INPUT.PIXEL_MEAN),
                RandomSampleCrop(),
                RandomMirror(),
                ToPercentCoords(),
                Resize(cfg.INPUT.IMAGE_SIZE),
                Remake(cfg.INPUT.IMAGE_SIZE),
                Standardize(cfg.INPUT.STANDARDIZATION_MEAN, cfg.INPUT.STANDARDIZATION_STDEV),
                ToTensor(),
            ]
        elif cfg.INPUT.AUTO_AUG == "v3":
            transform = [
                BbAug(cfg.INPUT.AUTO_AUG_VERSION),
                ConvertFromInts(),
                RandomSampleCrop(),
                RandomMirror(),
                ToPercentCoords(),
                Resize(cfg.INPUT.IMAGE_SIZE),
                Remake(cfg.INPUT.IMAGE_SIZE),
                Standardize(cfg.INPUT.STANDARDIZATION_MEAN, cfg.INPUT.STANDARDIZATION_STDEV),
                ToTensor(),
            ]
        elif cfg.INPUT.AUTO_AUG == "v4":
            transform = [
                BbAug(cfg.INPUT.AUTO_AUG_VERSION),
                ConvertFromInts(),
                PhotometricDistort(),
                RandomSampleCrop(),
                RandomMirror(),
                ToPercentCoords(),
                Resize(cfg.INPUT.IMAGE_SIZE),
                Remake(cfg.INPUT.IMAGE_SIZE),
                Standardize(cfg.INPUT.STANDARDIZATION_MEAN, cfg.INPUT.STANDARDIZATION_STDEV),
                ToTensor(),
            ]
        elif  cfg.INPUT.AUG == "v2":
            transform = [
                ConvertFromInts(),
                PhotometricDistort(),
                # Expand(cfg.INPUT.PIXEL_MEAN),
                RandomSampleCrop(),
                RandomMirror(),
                ToPercentCoords(),
                Resize(cfg.INPUT.IMAGE_SIZE),
                Remake(cfg.INPUT.IMAGE_SIZE),
                Standardize(cfg.INPUT.STANDARDIZATION_MEAN, cfg.INPUT.STANDARDIZATION_STDEV),
                ToTensor(),
            ]
        elif  cfg.INPUT.AUG == "v3":
            transform = [
                ConvertFromInts(),
                # PhotometricDistort(),
                Expand(cfg.INPUT.PIXEL_MEAN),
                RandomSampleCrop(),
                RandomMirror(),
                ToPercentCoords(),
                Resize(cfg.INPUT.IMAGE_SIZE),
                Remake(cfg.INPUT.IMAGE_SIZE),
                Standardize(cfg.INPUT.STANDARDIZATION_MEAN, cfg.INPUT.STANDARDIZATION_STDEV),
                ToTensor(),
            ]
        elif  cfg.INPUT.AUG == "v4":
            transform = [
                ConvertFromInts(),
                # PhotometricDistort(),
                # Expand(cfg.INPUT.PIXEL_MEAN),
                RandomSampleCrop(),
                RandomMirror(),
                ToPercentCoords(),
                Resize(cfg.INPUT.IMAGE_SIZE),
                Remake(cfg.INPUT.IMAGE_SIZE),
                Standardize(cfg.INPUT.STANDARDIZATION_MEAN, cfg.INPUT.STANDARDIZATION_STDEV),
                ToTensor(),
            ]
        else:
            transform = [
                ConvertFromInts(),
                PhotometricDistort(),
                Expand(cfg.INPUT.PIXEL_MEAN),
                RandomSampleCrop(),
                RandomMirror(),
                ToPercentCoords(),
                Resize(cfg.INPUT.IMAGE_SIZE),
                Remake(cfg.INPUT.IMAGE_SIZE),
                Standardize(cfg.INPUT.STANDARDIZATION_MEAN, cfg.INPUT.STANDARDIZATION_STDEV),
                ToTensor(),
            ]

    else:
        transform = [
            Resize(cfg.INPUT.IMAGE_SIZE),
            Remake(cfg.INPUT.IMAGE_SIZE),
            Standardize(cfg.INPUT.STANDARDIZATION_MEAN, cfg.INPUT.STANDARDIZATION_STDEV),
            ToTensor()
        ]
    transform = Compose(transform)
    return transform


def transform_NanoDet(cfg, is_train=True):
    """
    Data transforms for SSD box head
    :param cfg: config file
    :param is_train: parameter to differentiate between train and test mode
    :return: transformed data
    """
    if is_train:
        transform = [
            ConvertFromInts(),
            PhotometricDistort(),
            Expand(cfg.INPUT.PIXEL_MEAN),
            RandomSampleCrop(),
            RandomMirror(),
            ToPercentCoords(),
            Resize(cfg.INPUT.IMAGE_SIZE),
            Remake(cfg.INPUT.IMAGE_SIZE),
            SubtractMeans(cfg.INPUT.PIXEL_MEAN),
            ToTensor(),
        ]
    else:
        transform = [
            Resize(cfg.INPUT.IMAGE_SIZE),
            Remake(cfg.INPUT.IMAGE_SIZE),
            SubtractMeans(cfg.INPUT.PIXEL_MEAN),
            ToTensor()
        ]
    transform = Compose(transform)
    return transform

def transform_Yolo(cfg, is_train=True):
    """
    Data transforms for YOLO v2 and v3
    :param cfg: config file
    :param is_train: parameter to differentiate between train and test mode
    :return: transformed data
    """
    if is_train:
        transform = [
            RandomFlip(),
            YoloCrop(),
            ToPercentCoords(),
            Resize(cfg.INPUT.IMAGE_SIZE),
            SubtractMeans(cfg.INPUT.PIXEL_MEAN),
            ToTensor(),
        ]
    else:
        transform = [
            Resize(cfg.INPUT.IMAGE_SIZE),
            SubtractMeans(cfg.INPUT.PIXEL_MEAN),
            ToTensor()
        ]
    transform = Compose(transform)
    return transform


def transform_CenterNet(cfg, is_train=True):
    """
    Data transforms for CenterNetHead
    :param cfg: config file
    :param is_train: parameter to differentiate between train and test mode
    :return: transformed data
    """
    if is_train:
        transform = [
            ConvertFromInts(),
            PhotometricDistort(),
            RandomSampleCrop(),
            RandomMirror(),
            ToPercentCoords(),
            Resize(cfg.INPUT.IMAGE_SIZE),
            SubtractMeans(cfg.INPUT.PIXEL_MEAN),
            ToTensor(),
        ]
    else:
        transform = [
            Resize(cfg.INPUT.IMAGE_SIZE),
            SubtractMeans(cfg.INPUT.PIXEL_MEAN),
            ToTensor()
        ]
    transform = Compose(transform)
    return transform


def transform_EfficientDet(cfg, is_train=True):
    """
    Data transforms for EfficientDet
    :param cfg: config file
    :param is_train: parameter to differentiate between train and test mode
    :return: transformed data
    """
    if is_train:
        transform = [
            ConvertFromInts(),
            PhotometricDistort(),
            RandomSampleCrop(),
            RandomMirror(),
            ToPercentCoords(),
            Resize(cfg.INPUT.IMAGE_SIZE),
            ToAbsoluteCoords(),
            Standardize(cfg.INPUT.STANDARDIZATION_MEAN, cfg.INPUT.STANDARDIZATION_STDEV),
            ToTensor(),
        ]
    else:
        transform = [
            ToPercentCoords(),
            Resize(cfg.INPUT.IMAGE_SIZE),
            ToAbsoluteCoords(),
            Standardize(cfg.INPUT.STANDARDIZATION_MEAN, cfg.INPUT.STANDARDIZATION_STDEV),
            ToTensor()
        ]
    transform = Compose(transform)
    return transform


"""
Now, we write the main controller functions to call Head-Specific transforms 
"""


def build_transforms(cfg, is_train=True):
    """
    Image transforms (for all heads)
    :param cfg: config file
    :param is_train: train or test mode
    :return: return the transformations for the calling head
    """
    if "SSD" in cfg.MODEL.HEAD.NAME:
        return transform_SSD(cfg, is_train)
    elif "ThunderNet" in cfg.MODEL.HEAD.NAME:
        return transform_ThunderNet(cfg, is_train)
    elif "Yolov2Head" in cfg.MODEL.HEAD.NAME or "Yolov3Head" in cfg.MODEL.HEAD.NAME:
        return transform_Yolo(cfg, is_train)
    elif "CenterNetHead" in cfg.MODEL.HEAD.NAME:
        return transform_CenterNet(cfg, is_train)
    elif "FCOSHead" in cfg.MODEL.HEAD.NAME:
        return transform_FCOS(cfg, is_train)
    elif "EfficientDetHead" in cfg.MODEL.HEAD.NAME:
        return transform_EfficientDet(cfg, is_train)
    elif "NanoDetHead" in cfg.MODEL.HEAD.NAME:
        return transform_NanoDet(cfg, is_train)
    else:
        raise NotImplementedError("Transformation for detection head {} not implemented.".format(cfg.MODEL.HEAD.NAME))


def build_target_transform(cfg):
    """
    Target transforms (for all heads) - ground truth boxes and labels or heatmaps
    :param cfg: config file
    :return: return the transformations for the calling head
    """
    if "SSD" in cfg.MODEL.HEAD.NAME:
        from od.data.transforms.target_transform import SSDTargetTransform
        return SSDTargetTransform(PriorBox(cfg)(), cfg.MODEL.CENTER_VARIANCE, cfg.MODEL.SIZE_VARIANCE, cfg.MODEL.THRESHOLD)
    elif "ThunderNet" in cfg.MODEL.HEAD.NAME:
        from od.data.transforms.target_transform import ThunderNetTargetTransform
        return ThunderNetTargetTransform(cfg.INPUT.MAX_NUM_GT_BOXES)
    elif "Yolov2Head" in cfg.MODEL.HEAD.NAME or "Yolov3Head" in cfg.MODEL.HEAD.NAME:
        from od.data.transforms.target_transform import YoloTargetTransform
        return YoloTargetTransform(cfg.INPUT.MAX_NUM_GT_BOXES)
    elif "CenterNetHead" in cfg.MODEL.HEAD.NAME:
        from od.data.transforms.target_transform import CenterNetHeadTransform
        return CenterNetHeadTransform(cfg)
    elif "FCOSHead" in cfg.MODEL.HEAD.NAME:
        from od.data.transforms.target_transform import FCOSTargetTransform
        return FCOSTargetTransform(cfg.INPUT.IMAGE_SIZE)
    elif "EfficientDetHead" in cfg.MODEL.HEAD.NAME:
        from od.data.transforms.target_transform import EfficientDetTargetTransform
        return EfficientDetTargetTransform(cfg.INPUT.MAX_NUM_GT_BOXES)
    elif "NanoDetHead" in cfg.MODEL.HEAD.NAME:
        from od.data.transforms.target_transform import NanoDetTargetTransform
        return NanoDetTargetTransform()
    else:
        raise NotImplementedError(
            "Target transformation for detection head {} not implemented.".format(cfg.MODEL.HEAD.NAME))
