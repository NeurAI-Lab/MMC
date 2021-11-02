from yacs.config import CfgNode as CN

_C = CN(new_allowed=True)

_C.MODEL = CN(new_allowed=True)
# ---------------------------------------------------------------------------- #
# YOLO
# ---------------------------------------------------------------------------- #
_C.MODEL.NUM_ANCHORS = 5
_C.MODEL.ANCHORS = [1.3221, 1.73145, 3.19275, 4.00944, 5.05587, 8.09892, 9.47112, 4.84053, 11.2364, 10.0071]
_C.MODEL.LR_SCALES = [1, 5, 10, 0.1, 0.01]
_C.MODEL.OBJECT_SCALE = 5
_C.MODEL.NO_OBJECT_SCALE = 1
_C.MODEL.nms = 'ssd'