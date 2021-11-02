from od.modeling import registry


def build_head(cfg):
    head_name = cfg.MODEL.HEAD.NAME
    if head_name == 'SSDBoxHead':
        from od.modeling.head.ssd.box_head import SSDBoxHead
    elif head_name == 'Yolov2Head':
        from od.modeling.head.yolo.box_head import Yolov2Head
    elif head_name == 'Yolov3Head':
        from od.modeling.head.yolov3.box_head import Yolov3Head
    elif head_name == 'ThunderNetHead':
        from od.modeling.head.ThunderNet.thundernet.thundernet import ThunderNetHead
    elif head_name == 'CenterNetHead':
        from od.modeling.head.centernet.centernet_head import CenterNetHead
    elif head_name == 'FCOSHead':
        from od.modeling.head.fcos.fcos_head import FCOSHead
    elif head_name == 'EfficientDetHead':
        from od.modeling.head.efficient_det.box_head import EfficientDetHead
    elif head_name == 'NanoDetHead':
        from od.modeling.head.nanodet.nanodet_head import NanoDetHead
    else:
        raise ValueError(f"Undefined head: {head_name}")

    return registry.HEADS[cfg.MODEL.HEAD.NAME](cfg)
