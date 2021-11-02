from yacs.config import CfgNode as CN

_C = CN(new_allowed=True)

_C.MODEL = CN(new_allowed=True)
# -----------------------------------------------------------------------------
# PAN
# -----------------------------------------------------------------------------
_C.MODEL.PAN = CN(new_allowed=True)
_C.MODEL.PAN.OUT_CHANNELS = 96
# -----------------------------------------------------------------------------
# HEAD
# -----------------------------------------------------------------------------
_C.MODEL.HEAD = CN(new_allowed=True)
_C.MODEL.HEAD.FEAT_CHANNELS=96
_C.MODEL.HEAD.STACKED_CONVS=2
_C.MODEL.HEAD.SHARE_CLS_REG=True

_C.MODEL.HEAD.REG_MAX=7
_C.MODEL.HEAD.STRIDES=[8, 16, 32]
_C.MODEL.HEAD.NORM_CFG_TYPE= 'BN'
# -----------------------------------------------------------------------------
# LOSS
# -----------------------------------------------------------------------------
_C.MODEL.LOSS = CN(new_allowed=True)
_C.MODEL.LOSS.OCTAVE_BASE_SCALE=5
_C.MODEL.LOSS.SCALES_PER_OCTAVE=1
# ---------------------------------------------------------------------------- #
#
# ---------------------------------------------------------------------------- #
