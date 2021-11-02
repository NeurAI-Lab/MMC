from yacs.config import CfgNode as CN
INF = 100000000

_C = CN(new_allowed=True)
_C.MODEL = CN(new_allowed=True)



_C.MODEL = CN(new_allowed=True)
_C.MODEL.FPN_STRIDES = [8,16, 32, 64, 128]
_C.MODEL.PRIOR_PROB = 0.01

_C.MODEL.PRETRAINED_PATH = "/input/datasets/uninet/pytorch-config/FCOS_imprv_R_50_FPN_1x.pth"
# Focal loss parameter: alpha
_C.MODEL.LOSS_ALPHA = 0.25
# Focal loss parameter: gamma
_C.MODEL.LOSS_GAMMA = 2.0

# the number of convolutions used in the cls and bbox tower
_C.MODEL.NUM_CONVS = 4

# if CENTER_SAMPLING_RADIUS <= 0, it will disable center sampling
_C.MODEL.CENTER_SAMPLING_RADIUS = 0.0
# IOU_LOSS_TYPE can be "iou", "linear_iou" or "giou"
_C.MODEL.IOU_LOSS_TYPE = "iou"

_C.MODEL.NORM_REG_TARGETS = False
_C.MODEL.CENTERNESS_ON_REG = False

_C.MODEL.USE_DCN_IN_TOWER = False
_C.MODEL.ENCODER="Simple"
_C.MODEL.ENCODER_BATCH_NORM=False
_C.MODEL.FPN = CN(new_allowed=True)
_C.MODEL.FPN.USE_GN = False
_C.MODEL.FPN.USE_RELU = False
_C.MODEL.FPN.OUT_CHANNELS=512
_C.MODEL.OBJECT_SIZES_OF_INTEREST=[
            [-1, 64],
            [64, 128],
            [128, 256],
            [256, 512],
            [512, INF],
        ]
_C.MODEL.GROUP_NORM = CN(new_allowed=True)
# Number of dimensions per group in GroupNorm (-1 if using NUM_GROUPS)
_C.MODEL.GROUP_NORM.DIM_PER_GP = -1
# Number of groups in GroupNorm (-1 if using DIM_PER_GP)
_C.MODEL.GROUP_NORM.NUM_GROUPS = 32
# GroupNorm's small constant in the denominator
_C.MODEL.GROUP_NORM.EPSILON = 1e-5
_C.TEST = CN(new_allowed=True)
_C.TEST.PRE_NMS_TOP_N = 1000
_C.TEST.BBOX_AUG_ENABLED = False


