from .defaults import _C as cfg

try:
    from .ssd_config import _C as ssd_cfg
except ImportError:
    pass

try:
    from .centernet_config import _C as centernet_cfg
except ImportError:
    pass

try:
    from .thundernet_config import _C as thundernet_cfg
except ImportError:
    pass

try:
    from .yolo_config import _C as yolo_cfg
except ImportError:
    pass

try:
    from .fcos_config import _C as fcos_cfg
except ImportError:
    pass

try:
    from .efficientdet_config import _C as efficientdet_cfg
except ImportError:
    pass
try:
    from .nanodet_config  import _C as nanodet_cfg
except ImportError:
    pass
