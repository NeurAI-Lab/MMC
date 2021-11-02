import logging
import os
from datetime import datetime

import numpy as np

from .eval_detection_voc import eval_detection_voc
from od.utils.box_utils import plot_pred_gt

def voc_evaluation(cfg, dataset, predictions, output_dir, iteration=None, precision_display=False, iou_threshold=0.5, dataset_type=None):
    class_names = dataset.class_names

    pred_boxes_list = []
    pred_labels_list = []
    pred_scores_list = []
    gt_boxes_list = []
    gt_labels_list = []
    gt_difficults = []

    for i in range(len(dataset)):
        image_id, annotation = dataset.get_annotation(i)
        gt_boxes, gt_labels, is_difficult = annotation
        gt_boxes_list.append(gt_boxes)
        gt_labels_list.append(gt_labels)
        gt_difficults.append(is_difficult.astype(np.bool))

        img_info = dataset.get_img_info(i)

        prediction = predictions[i]
        prediction = prediction.resize((img_info['width'], img_info['height'])).numpy()
        boxes, labels, scores = prediction['boxes'], prediction['labels'], prediction['scores']

        pred_boxes_list.append(boxes)
        pred_labels_list.append(labels)
        pred_scores_list.append(scores)

        if cfg.LOGGER.DEBUG_MODE:
            # ---------Utility - Display Ground Truth-----------
            save_dir = "Det"
            plot_pred_gt(dataset, image_id, gt_boxes, gt_labels, boxes, labels, save_dir)

    result = eval_detection_voc(pred_bboxes=pred_boxes_list,
                                pred_labels=pred_labels_list,
                                pred_scores=pred_scores_list,
                                gt_bboxes=gt_boxes_list,
                                gt_labels=gt_labels_list,
                                gt_difficults=gt_difficults,
                                iou_thresh=0.5,
                                use_07_metric=True)
    logger = logging.getLogger("Object Detection.inference")
    if precision_display:
        result_str = 'mAP: {:.4f}, mAR: {:.4f}, mF1: {:.4f}\n'.format(result["map"], result["mar"], result["mf1"])
    else:
        result_str = "mAP: {:.4f}\n".format(result["map"])
    metrics = {'mAP': result["map"]}
    print('MAP : %f' % result["map"])
    for i, (ar, f1, ap) in enumerate(zip(result["ar"], result["f1"], result["ap"])):
        if(cfg.DATA_LOADER.INCLUDE_BACKGROUND):
            if i == 0:  # skip background
                continue
        if precision_display:
            metrics[class_names[i]] = ar
            metrics[class_names[i]] = f1
        metrics[class_names[i]] = ap
        if precision_display:
            result_str += "Category: {:<16}: AP: {:.4f}, AR: {:.4f}, F1-score: {:.4f}\n".format(class_names[i], ap, ar, f1)
        else:
            result_str += "{:<16}: {:.4f}\n".format(class_names[i], ap)
    logger.info(result_str)

    if iteration is not None:
        result_path = os.path.join(output_dir, 'result_{:07d}.txt'.format(iteration))
    else:
        result_path = os.path.join(output_dir, 'result_{}.txt'.format(datetime.now().strftime('%Y-%m-%d_%H-%M-%S')))
    with open(result_path, "w") as f:
        f.write(result_str)

    return dict(metrics=metrics)
