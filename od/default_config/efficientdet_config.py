from yacs.config import CfgNode as CN

_C = CN(new_allowed=True)
_C.MODEL = CN(new_allowed=True)
_C.MODEL.HEAD = CN(new_allowed=True)
_C.MODEL.HEAD.BIFPN = CN(new_allowed=True)
_C.MODEL.HEAD.RETINAHEAD = CN(new_allowed=True)

# ---------------------------------------------------------------------------- #
# EfficientDet setting specific for each head/backbone in the family
# ---------------------------------------------------------------------------- #

# Head name
_C.MODEL.HEAD.NAME = "EfficientDetHead"

# Use pretrained head?
_C.MODEL.HEAD.PRETRAINED = True

# Path to pretrained weights of the head?
_C.MODEL.HEAD.WEIGHTS_PATH = ""

# With @INPUT_CHANNELS we specify how deep is are the feature levels
# passed to BiFPN
_C.MODEL.HEAD.BIFPN.INPUT_CHANNELS = [-1]

# With @WIDTH we specify the number of channels of BIFPN network
_C.MODEL.HEAD.BIFPN.WIDTH = -1

# With @DEPTH we specify the number of layers of BIFPN network
_C.MODEL.HEAD.BIFPN.DEPTH = -1

# With @DEPTH we specify the number of layers of head box/cls
# prediction network
_C.MODEL.HEAD.RETINAHEAD.DEPTH = -1

# ---------------------------------------------------------------------------- #
# EfficientDet BIFPN default setting common for all heads in the family
# ---------------------------------------------------------------------------- #

# With @PYRAMID_LEVELS we specify which levels do we take as an input to
# BiFPN stage - P3-P7
_C.MODEL.HEAD.BIFPN.PYRAMID_LEVELS = [3, 4, 5, 6, 7]

# With @USE_DEPTH_SEPARABLE_CONV we specify if the BiFPN network should
# use depth separable convolution instead of normal 2DConv
_C.MODEL.HEAD.BIFPN.USE_DEPTH_SEPARABLE_CONV = True

# With @BATCH_NORM we specify if we want to apply batch norm in BiFPN
# network (only standard BN is supported at the moment)
_C.MODEL.HEAD.BIFPN.BATCH_NORM = True

# With @BATCH_NORM_DECAY we specify batch norm weight decay
_C.MODEL.HEAD.BIFPN.BATCH_NORM_DECAY = 0.99

# With @BATCH_NORM_EPS we specify batch norm eps
_C.MODEL.HEAD.BIFPN.BATCH_NORM_EPS = 1e-3

# With @ACTIVATION we specify activation function to be used in BiFPN
# network - currently supported: 'relu', 'swish', 'sigmoid' and None
_C.MODEL.HEAD.BIFPN.ACTIVATION = 'swish'

# With @FAST_ATTENTION we specify the way of feature fusion in BiFPN.
# For EfficientDet-B0-5 fast attention is set true, but for two biggest
# models (B6/B7) it's set to False.
_C.MODEL.HEAD.BIFPN.FAST_ATTENTION = True

# ---------------------------------------------------------------------------- #
# EfficientDet RetinaHead default setting
# ---------------------------------------------------------------------------- #

# Focal loss alpha coeff
_C.MODEL.HEAD.RETINAHEAD.FOCAL_LOSS_ALPHA = 0.25

# Focal loss gamma coeff
_C.MODEL.HEAD.RETINAHEAD.FOCAL_LOSS_GAMMA = 1.5

# Huber delta coefficient
_C.MODEL.HEAD.RETINAHEAD.HUBER_DELTA = 0.1

# Balance classification loss and bounding box loss with @BOX_LOSS_WEIGHT
# multiplier
_C.MODEL.HEAD.RETINAHEAD.BOX_LOSS_WEIGHT = 50

# With @USE_DEPTH_SEPARABLE_CONV we specify if the retina head
# network should use depth separable convolution instead of normal 2DConv
_C.MODEL.HEAD.RETINAHEAD.USE_DEPTH_SEPARABLE_CONV = True

# With @BATCH_NORM we specify if we want to apply batch norm in BiFPN
# network (only standard BN is supported at the moment)
_C.MODEL.HEAD.RETINAHEAD.BATCH_NORM = True

# With @\BATCH_NORM_DECAY we specify batch norm weight decay
_C.MODEL.HEAD.RETINAHEAD.BATCH_NORM_DECAY = 0.99

# With @BATCH_NORM_EPS we specify batch norm eps
_C.MODEL.HEAD.RETINAHEAD.BATCH_NORM_EPS = 1e-3

# With @ACTIVATION we specify activation function to be used in retina
# head network - currently supported: 'relu', 'swish', 'sigmoid' and None
_C.MODEL.HEAD.RETINAHEAD.ACTIVATION = 'swish'

# With @MIN_CONFIDENCE we specify the minimum confidance of
# a bbox to be considered by retina head
_C.MODEL.HEAD.RETINAHEAD.MIN_CONFIDENCE = 0.01
