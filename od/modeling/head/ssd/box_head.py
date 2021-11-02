from torch import nn
from torch import autograd
import torch.nn.functional as F

from od.modeling import registry
from od.modeling.anchors.prior_box import PriorBox
from od.modeling.head.ssd.box_predictor import make_box_predictor
from od.utils import box_utils
from od.modeling.head.ssd.inference import PostProcessor
from od.modeling.head.ssd.loss import MultiBoxLoss

class NMSWithOnnxSupport(nn.Module):
    def __init__(self, cpu_nms, confidence_threshold, nms_threshold, max_per_class, max_per_image):
        super(NMSWithOnnxSupport, self).__init__()

        class OnnxNMS(autograd.Function):
            confidence_threshold = None
            nms_threshold = None
            max_per_class = None
            max_per_image = None
            cpu_nms = None

            @staticmethod
            def forward(ctx, scores, boxes):
                # As we reshaped earlier, reshape back to original size
                out = cpu_nms([scores, boxes.reshape([1,-1,4])])

                # ONNX does not support the Container class. We must unpack the
                # output tensors.
                boxes = out[0]['boxes']
                labels = out[0]['labels']
                scores = out[0]['scores']

                # While exporting onnx, we are not allowed to return multiple tensors.
                # Following would cause a crash:
                # return boxes, labels, scores
                # Unfortunately this breaks the normal of the program.
                return scores

            @staticmethod
            def symbolic(g, scores, boxes):
                return g.op('nms_TRT', scores, boxes,
                            confidence_threshold_f=OnnxNMS.confidence_threshold,
                            nms_threshold_f=OnnxNMS.nms_threshold,
                            max_per_image_i=OnnxNMS.max_per_image,
                            max_per_class_i=OnnxNMS.max_per_class
                            )

        OnnxNMS.cpu_nms = cpu_nms
        OnnxNMS.confidence_threshold = confidence_threshold
        OnnxNMS.max_per_class = max_per_class
        OnnxNMS.max_per_image = max_per_image
        OnnxNMS.nms_threshold = nms_threshold

        self.implementation = OnnxNMS()

    def forward(self, scores_boxes):
        scores = scores_boxes[0]
        boxes = scores_boxes[1]

        # Reshaping is easiest to do here as ONNX complicates things.
        # Basically reshaping in the symbolic static method is not possible / easy

        boxes = boxes.reshape([1, -1, 1, 4])
        res = self.implementation.apply(scores, boxes)
        return res


@registry.HEADS.register('SSDBoxHead')
class SSDBoxHead(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.predictor = make_box_predictor(cfg)
        self.loss_evaluator = MultiBoxLoss(neg_pos_ratio=cfg.MODEL.NEG_POS_RATIO, num_classes=self.cfg.MODEL.NUM_CLASSES, focal_loss=cfg.MODEL.HEAD.FOCAL_LOSS)
        self.post_processor = PostProcessor(cfg)
        self.priors = None
        if self.cfg.EXPORT == 'onnx':
            self.nms = NMSWithOnnxSupport(self.post_processor,
                                          confidence_threshold=cfg.TEST.CONFIDENCE_THRESHOLD,
                                          nms_threshold=cfg.TEST.NMS_THRESHOLD,
                                          max_per_class=cfg.TEST.MAX_PER_CLASS,
                                          max_per_image=cfg.TEST.MAX_PER_IMAGE)

    def forward(self, features, targets=None, teacher=False):
        cls_logits, bbox_pred = self.predictor(features)
        if self.training:
            return self._forward_train(cls_logits, bbox_pred, targets)
        else:
            return self._forward_test(cls_logits, bbox_pred)

    def _forward_train(self, cls_logits, bbox_pred, targets):
        gt_boxes, gt_labels = targets['boxes'], targets['labels']
        # reg_loss, cls_loss = self.loss_evaluator(cls_logits, bbox_pred, gt_labels, gt_boxes)
        # loss_dict = dict(
        #     reg_loss=reg_loss,
        #     cls_loss=cls_loss,
        # )
        detections = (cls_logits, bbox_pred)
        return detections

    def _forward_test(self, cls_logits, bbox_pred):
        if self.priors is None:
            self.priors = PriorBox(self.cfg)().to(bbox_pred.device)
        scores = F.softmax(cls_logits, dim=2)
        boxes = box_utils.convert_locations_to_boxes(
            bbox_pred, self.priors, self.cfg.MODEL.CENTER_VARIANCE, self.cfg.MODEL.SIZE_VARIANCE
        )
        boxes = box_utils.center_form_to_corner_form(boxes)
        detections = (scores, boxes)
        if self.cfg.EXPORT == 'onnx':
            detections = self.nms(detections)
        else:
            detections = self.post_processor(detections)
        return detections, {}
