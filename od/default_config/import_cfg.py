from od.default_config import ssd_cfg
from od.default_config import thundernet_cfg
from od.default_config import yolo_cfg
from od.default_config import centernet_cfg
from od.default_config import efficientdet_cfg
from od.default_config import fcos_cfg
from od.default_config import nanodet_cfg
from od.default_config import cfg

sub_cfg_dict = {"SSDBoxHead": ssd_cfg, "ThunderNetHead": thundernet_cfg, "Yolov2Head": yolo_cfg, "Yolov3Head": yolo_cfg,
                "CenterNetHead": centernet_cfg, "FCOSHead": fcos_cfg, "EfficientDetHead": efficientdet_cfg,"NanoDetHead":nanodet_cfg}