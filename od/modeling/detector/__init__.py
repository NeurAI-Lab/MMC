from .TwoPhasesDetector import TwoPhasesDetector

_DETECTION_META_ARCHITECTURES = {
    "TwoPhasesDetector": TwoPhasesDetector,
}


def build_detection_model(cfg):
    meta_arch = _DETECTION_META_ARCHITECTURES[cfg.MODEL.META_ARCHITECTURE]
    return meta_arch(cfg)
