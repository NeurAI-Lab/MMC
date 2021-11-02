import logging
import os
import copy
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.utils.data
from torch.autograd import Variable
from tqdm import tqdm
import time
import math
import numpy as np

from od.data.build import make_data_loader, make_test_data_loader
from od.data.datasets.evaluation import evaluate

from od.utils import dist_util, mkdir
from od.utils.dist_util import synchronize, is_main_process
from od.utils.adverserial_utils import clamp_tensor, forward_transform

# from torchsummary import summary


def _accumulate_predictions_from_multiple_gpus(predictions_per_gpu):
    all_predictions = dist_util.all_gather(predictions_per_gpu)
    if not dist_util.is_main_process():
        return
    # merge the list of dicts
    predictions = {}
    for p in all_predictions:
        predictions.update(p)
    # convert a dict where the key is the index in a list
    image_ids = list(sorted(predictions.keys()))
    if len(image_ids) != image_ids[-1] + 1:
        logger = logging.getLogger("Object Detection.inference")
        logger.warning(
            "Number of images that were gathered from multiple processes is not "
            "a contiguous set. Some images might be missing from the evaluation"
        )

    # convert to a list
    predictions = [predictions[i] for i in image_ids]
    return predictions


@torch.enable_grad()
def pgd_attack(cfg, model, images, targets, eps=0, step_size=1/255.):
    # setting iters based on epsilon
    iters = int(min(eps+4.0, math.ceil(1.25*eps)))

    # normalizing the epsilon value
    eps = eps/255.0

    # change targets from one class to other
    # targeted attack step (change targets)
    for i in range(len(targets['labels'])):
        if targets['labels'][0][i] == 1:
            #  if person, change it to background
            targets['labels'][0][i] = 0

    # from https://github.com/columbia/MTRobust/blob/99d17939161fd7425cba5f32472ca85db0cace64/learning/attack.py
    tensor_std = images.std(dim=(0,2,3))
    tensor_std = tensor_std.unsqueeze(0)
    tensor_std = tensor_std.unsqueeze(2)
    tensor_std = tensor_std.unsqueeze(2).float()

    x_adv = images.clone()

    pert_epsilon = torch.ones_like(x_adv) * eps / tensor_std
    pert_upper = x_adv + pert_epsilon
    pert_lower = x_adv - pert_epsilon

    upper_bound = torch.ones_like(x_adv)
    lower_bound = torch.zeros_like(x_adv)

    info = {}
    info["mean"] = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32)
    info["std"] = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32)

    upper_bound = forward_transform(upper_bound, info)
    lower_bound = forward_transform(lower_bound, info)

    upper_bound = torch.min(upper_bound, pert_upper)
    lower_bound = torch.max(lower_bound, pert_lower)

    ones_x = torch.ones_like(images).float()

    x_adv = x_adv.cuda()
    upper_bound = upper_bound.cuda()
    lower_bound = lower_bound.cuda()
    tensor_std = tensor_std.cuda()
    ones_x = ones_x.cuda()

    device = torch.device('cuda')
    targets = targets.to(device)
    step_size_tensor = ones_x * step_size / tensor_std

    noise = torch.FloatTensor(images.size()).uniform_(-eps, eps)
    noise = noise.cuda()
    noise = noise / tensor_std
    x_adv = x_adv + noise
    x_adv = clamp_tensor(x_adv, lower_bound, upper_bound)

    x_adv.requires_grad = True

    model.training = True

    # checking for detection head or segmentation head

    for i in range(iters):
        images.requires_grad = True
        #loss_dict, _ = model(x_adv.to(device), targets=targets)
        model.train()  # *IMPORTANT*: change to train mode after eval.
        detections = model(x_adv.to(device), targets=targets)
        from od.modeling.detector.head_loss import ssd_loss
        reg_loss, cls_loss = ssd_loss(cfg, detections, targets=targets)
        loss_dict = dict(
            reg_loss=reg_loss,
            cls_loss=cls_loss,
        )
        model.zero_grad()
        if x_adv.grad is not None:
            x_adv.grad.data.fill_(0)

        loss_dict["cls_loss"].backward()

        x_adv.grad.sign_()
        # change direction (NEGATIVE) for targeted attack
        x_adv = x_adv - (step_size_tensor * x_adv.grad)
        x_adv = clamp_tensor(x_adv, upper_bound, lower_bound)
        x_adv = Variable(x_adv.data, requires_grad=True)

    model.training = False
    model.eval()

    return x_adv


def compute_on_dataset(model, data_loader, device, cfg, epsilon, output_folder=None):

    results_dict = {}
    cpu_device = torch.device("cpu")

    for batch in tqdm(data_loader):
        images, targets, image_ids = batch

        if targets is not None:
            images = pgd_attack(cfg, model, images, targets, epsilon)

        with torch.no_grad():
            outputs = model(images.to(device))

            outputs = [o.to(cpu_device) for o in outputs]
            results_dict.update(
                {img_id: result for img_id, result in zip(image_ids, outputs)}
            )

    return results_dict


def get_inference_time(model, data_loader, device, test_it=501):
    run_time = []
    data_sampler = iter(data_loader)

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    with torch.no_grad():
        for i in range(test_it):
            try:
                images, targets, image_ids = next(data_sampler)
            except StopIteration:
                batch_iterator = iter(data_loader)
                images, targets, image_ids = next(batch_iterator)
            images = images.to(device)
            torch.cuda.synchronize()

            start.record()
            outputs = model(images)
            end.record()

            torch.cuda.synchronize()
            run_time.append(start.elapsed_time(end))

    run_time = run_time[1:]
    avg_run_time = np.mean(run_time)

    return avg_run_time

def inference(cfg, model, data_loader, dataset_name, device, get_inf_time=False,
              epsilon=None, output_folder=None, use_cached=False, **kwargs):

    dataset = data_loader.dataset
    logger = logging.getLogger("Object Detection.inference")
    logger.info("Evaluating {} dataset({} images):".format(dataset_name, len(dataset)))
    predictions_path = os.path.join(output_folder, 'predictions.pth')

    inf_time = 0

    if get_inf_time:
        inf_time = get_inference_time(model, data_loader, device)
        print("Model Inference Time:", inf_time, "ms")

    if use_cached and os.path.exists(predictions_path):
        predictions = torch.load(predictions_path, map_location='cpu')
    else:
        predictions = compute_on_dataset(model, data_loader, device, cfg, epsilon, output_folder)
        synchronize()

        predictions = _accumulate_predictions_from_multiple_gpus(predictions)

    if not is_main_process():
        return
    if output_folder:
        torch.save(predictions, predictions_path)

    det_result = evaluate(cfg, dataset=dataset, predictions=predictions, output_dir=output_folder, **kwargs)

    return inf_time, det_result


@torch.no_grad()
def do_evaluation(cfg, model, distributed, epsilon=1, get_inf_time=False, **kwargs):
    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        model = model.module
    model.eval()
    device = torch.device(cfg.MODEL.DEVICE)

    data_loaders_val = make_test_data_loader(cfg, is_train=False, distributed=distributed)
    eval_results = []
    inf_speed_results = []
    # print("Evaluating for: {}".format(epsilon))
    for dataset_name, data_loader in zip(cfg.DATASETS.TEST, data_loaders_val):
    for dataset_name, data_loader in zip(cfg.DATASETS.TEST, data_loaders_val):
        # output_folder = os.path.join(cfg.OUTPUT_DIR, "e"+str(epsilon), "inference", dataset_name)
        output_folder = os.path.join(cfg.OUTPUT_DIR, "inference", dataset_name)
        if not os.path.exists(output_folder):
            mkdir(output_folder)
        inf_time, eval_result = inference(cfg, model, data_loader, dataset_name, device, get_inf_time,
                                                      output_folder=output_folder, epsilon=epsilon, **kwargs)
        inf_speed_results.append(inf_time)
        eval_results.append(eval_result)
    return inf_speed_results, eval_results