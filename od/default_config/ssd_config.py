from yacs.config import CfgNode as CN

_C = CN(new_allowed=True)

_C.MODEL = CN(new_allowed=True)
# -----------------------------------------------------------------------------
# PRIORS
# -----------------------------------------------------------------------------
_C.MODEL.PRIORS = CN(new_allowed=True)
_C.MODEL.PRIORS.STRIDES = [8, 16, 32, 64, 100, 300]
_C.MODEL.PRIORS.MIN_SIZES = [30, 60, 111, 162, 213, 264]
_C.MODEL.PRIORS.MAX_SIZES = [60, 111, 162, 213, 264, 315]
_C.MODEL.PRIORS.ASPECT_RATIOS = [[2], [2, 3], [2, 3], [2, 3], [2], [2]]
# When has 1 aspect ratio, every location has 4 boxes, 2 ratio 6 boxes.
# #boxes = 2 + #ratio * 2
_C.MODEL.PRIORS.BOXES_PER_LOCATION = [4, 6, 6, 6, 4, 4]  # number of boxes per feature map location
_C.MODEL.PRIORS.CLIP = True

# ---------------------------------------------------------------------------- #
#
# ---------------------------------------------------------------------------- #
_C.MODEL.BOX_PREDICTOR = 'SSDBoxPredictor'
