import json
import math
import logging
import os
from datetime import datetime
import numpy as np
import sdk.src.utils as utils
from od.utils.box_utils import plot_pred_gt
import cv2
from od.data.datasets.dataset_class_names import dataset_classes

def draw_detections(output_dir, dataset, coco_dt):
    for i in range(len(dataset.ids)):
        imgid = dataset.ids[i]
        # v = dataset.coco.imgToAnns[imgid]
        v = coco_dt.imgToAnns[imgid]
        k = coco_dt.imgs[imgid]["file_name"].split("/")[-1]
        img1 = cv2.imread(dataset.get_image_path(i))
        out=os.path.join(output_dir, k)
        for ii in v:
            if ii["score"]<.13:
                continue
            x1, y1, w, h = ii['bbox']
            x1, y1, x2, y2 = int(x1), int(y1), int(x1 + w), int(y1 + h)
            cat = ii['category_id']
            color = ((10 * cat) % 256, (20 * cat) % 256, (5 * cat) % 256)
            cv2.rectangle(img1, (x1, y1), (x2, y2), color, 2)
            text = rf"{dataset_classes['coco'][dataset.coco_id_to_contiguous_id[ii['category_id']]+1]}"
            cv2.putText(img1, text, (x1+8, y1+12), cv2.FONT_HERSHEY_SIMPLEX, .4, color,thickness=1, lineType=cv2.LINE_AA)
        cv2.imwrite(out, img1)

def calculate_map_at_iou(coco_eval, iou_th):
    precision = coco_eval.eval['precision']
    recall = coco_eval.eval['recall']

    params = coco_eval.params
    aRng_ind = [i for i, aRng in enumerate(params.areaRngLbl) if aRng == 'all']
    mDet_ind = [i for i, mDet in enumerate(params.maxDets) if mDet == 100]

    mAP = precision[np.where(iou_th == params.iouThrs)[0]]
    mAP = mAP[:, :, :, aRng_ind, mDet_ind]
    mAP = np.mean(mAP[mAP > -1])

    mAR = recall[np.where(iou_th == params.iouThrs)[0]]
    mAR = mAR[:, :, aRng_ind, mDet_ind]
    mAR = np.mean(mAR[mAR > -1])
    return mAP, mAR

def log_average_miss_rate(prec, rec):
    """
        log-average miss rate:
            Calculated by averaging miss rates at 9 evenly spaced FPPI points
            between 10e-2 and 10e0, in log-space.
        output:
                lamr | log-average miss rate
                mr | miss rate
                fppi | false positives per image
        references:
            [1] Dollar, Piotr, et al. "Pedestrian Detection: An Evaluation of the
               State of the Art." Pattern Analysis and Machine Intelligence, IEEE
               Transactions on 34.4 (2012): 743 - 761.
    """

    # if there were no detections of that class
    if prec.size == 0:
        lamr = 0
        mr = 1
        fppi = 0
        return lamr, mr, fppi

    fppi = (1 - prec)
    mr = (1 - rec)

    fppi_tmp = np.insert(fppi, 0, -1.0)
    mr_tmp = np.insert(mr, 0, 1.0)

    # Use 9 evenly spaced reference points in log-space
    ref = np.logspace(-2.0, 0.0, num = 9)
    for i, ref_i in enumerate(ref):
        # np.where() will always find at least 1 index, since min(ref) = 0.01 and min(fppi_tmp) = -1.0
        j = np.where(fppi_tmp <= ref_i)[-1][-1]
        ref[i] = mr_tmp[j]

    # log(0) is undefined, so we use the np.maximum(1e-10, ref)
    lamr = math.exp(np.mean(np.log(np.maximum(1e-10, ref))))

    return lamr, mr, fppi

bdd_class_mapper = {1: 1, 2:0, 3: 3, 4:6, 5:0, 6:0, 7:0, 8:0, 9:0, 10:0, 17:0, 18:0}

def coco_evaluation(cfg, dataset, predictions, output_dir, iteration=None, lamr=False, precision_display=False,
                    iou_threshold=0.5, dataset_type="coco"):
    coco_results = []
    for i, prediction in enumerate(predictions):
        img_info = dataset.get_img_info(i)
        prediction = prediction.resize((img_info['width'], img_info['height'])).numpy()
        boxes, labels, scores = prediction['boxes'], prediction['labels'], prediction['scores']

        image_id, annotation = dataset.get_annotation(i)
        if "BDD" in dataset.data_dir:
            class_mapper = {1: 0, 2: 1, 3: 2, 4: 3, 5: 4, 6: 5, 7: 6, 8: 7, 9: 8, 10: 9, 11: 10, 18: 17}
        else:
            class_mapper = dataset.contiguous_id_to_coco_id
        if labels.shape[0] == 0:
            continue

        #######################################################################
        # Debug mode - produce images with detections
        #######################################################################

        if cfg.LOGGER.DEBUG_MODE:
            save_dir = "Det"
            labels_mapped = [class_mapper[label] for label in labels]
            annotation_mapped = [class_mapper[a] for a in annotation[1]]
            if "BDD" in dataset.data_dir:
                labels_mapped = [bdd_class_mapper[label] for label in labels_mapped]
                annotation_mapped = [bdd_class_mapper[a] for a in annotation_mapped]
            plot_pred_gt(dataset, image_id, annotation[0], annotation_mapped, boxes, labels_mapped, save_dir, voc=False)

        boxes = boxes.tolist()
        labels = labels.tolist()
        scores = scores.tolist()
        coco_results.extend(
            [
                {
                    "image_id": image_id,
                    "category_id": bdd_class_mapper[class_mapper[labels[k]]] if "BDD" in dataset.data_dir else class_mapper[labels[k]],
                    "bbox": [box[0], box[1], box[2] - box[0], box[3] - box[1]],  # to xywh format
                    "score": scores[k],
                }
                for k, box in enumerate(boxes)
            ]
        )
    iou_type = 'bbox'
    json_result_file = os.path.join(output_dir, iou_type + ".json")
    logger = logging.getLogger("Object Detection.inference")
    logger.info('Writing results to {}...'.format(json_result_file))
    with open(json_result_file, "w") as f:
        json.dump(coco_results, f)
    from pycocotools.cocoeval import COCOeval
    coco_gt = dataset.coco
    coco_dt = coco_gt.loadRes(json_result_file)
    # draw_detections(output_dir,dataset, coco_dt)
    coco_eval = COCOeval(coco_gt, coco_dt, iou_type)
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    result_strings = []
    keys = ["AP", "AP50", "AP75", "APs", "APm", "APl"]
    metrics = {}
    for i, key in enumerate(keys):
        metrics[key] = coco_eval.stats[i]
        # rounding off to 4 decimal places for consistency with VOC
        logger.info('{:<10}: {}'.format(key, round(coco_eval.stats[i], 4)))
        result_strings.append('{:<10}: {}'.format(key, round(coco_eval.stats[i], 4)))
    # Calculate mAP, mAR at a fixed IOU
    iou_th = iou_threshold
    mAP_iou, mAR_iou = calculate_map_at_iou(coco_eval, iou_th)
    metrics["AP{}".format(iou_th)] = mAP_iou
    metrics["AR{}".format(iou_th)] = mAR_iou
    result_strings.append('AP{:<10}: {}'.format(iou_th, round(mAP_iou, 4)))
    logger.info('AP{:<10}: {}'.format(iou_th, round(mAP_iou, 3)))
    result_strings.append('AR{:<10}: {}'.format(iou_th, round(mAR_iou, 4)))
    logger.info('AR{:<10}: {}'.format(iou_th, round(mAR_iou, 3)))
    metrics["Avg Recall"] = -1
    metrics["F1-score"] = -1

    if precision_display:
        # addition for precision display to display class wise AP, AR and F1
        class_names = list(utils.get_class_names(dataset_type))
        if not cfg.DATA_LOADER.INCLUDE_BACKGROUND:
            class_names.remove("__background__")

        list_ap = []
        list_ar = []
        list_f1 = []
        total_ap, total_ar = calculate_map_at_iou(coco_eval, iou_th)
        total_f1 = 2 * ((total_ap * total_ar) / (total_ap + total_ar))
        list_ap.append(total_ap)
        list_ar.append(total_ar)
        list_f1.append(total_f1)

        for i in range(len(class_names)):
            if class_names[i] == "__background__":
                continue
            coco_eval.params.catIds = class_mapper[i]
            if "BDD" in dataset.data_dir:
                coco_eval.params.catIds = bdd_class_mapper[class_mapper[i]]
            coco_eval.evaluate()
            coco_eval.accumulate()
            coco_eval.summarize()
            ap_val, ar_val = calculate_map_at_iou(coco_eval, iou_th)
            list_ap.append(ap_val)
            list_ar.append(ar_val)
            list_f1.append(2 * ((ap_val * ar_val) / (ap_val + ar_val)))

        classes = ['Total'] + class_names
        if '__background__' in class_names:
            classes.remove('__background__')
        lamr, mr, fppi = log_average_miss_rate(np.array(list_ap), np.array(list_ar))

        for obj_cat, ap, ar, f1 in zip(classes, list_ap, list_ar, list_f1):
            print("Category: {:<16}: AP: {:.4f}, AR: {:.4f}, F1-score: {:.4f}".format(obj_cat, ap, ar, f1))
            result_strings.append("Category: {:<16}: AP: {:.4f}, AR: {:.4f}, F1-score: {:.4f}".format(obj_cat, ap, ar, f1))

            # print('Class: {0}, Precision: {1}, Recall: {2}, F1-Score: {3}'.format(obj_cat, ap_val, ar_val, f1_val))
        result_strings.append('{:<10}: {:.4f}'.format("Avg Recall", round(list_ar[0], 3)))
        result_strings.append('{:<10}: {:.4f}'.format("F1-Score", round(list_f1[0], 3)))
        result_strings.append('{:<10}: {:.4f}'.format("LAMR", round(lamr, 3)))
        metrics["Avg Recall"] = list_ar[0]
        metrics["F1-score"] = list_f1[0]
        metrics["LAMR"] = lamr
        print("LAMR: {:.4f}".format(lamr))

    if iteration is not None:
        result_path = os.path.join(output_dir, 'result_{:07d}.txt'.format(iteration))
    else:
        result_path = os.path.join(output_dir, 'result_{}.txt'.format(datetime.now().strftime('%Y-%m-%d_%H-%M-%S')))
    with open(result_path, "w") as f:
        f.write('\n'.join(result_strings))

    return dict(metrics=metrics)
