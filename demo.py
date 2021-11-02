import glob
import os
import time
import yaml

import torch
from PIL import Image
from vizer.draw import draw_boxes

from od.default_config import cfg
from od.data.datasets.dataset_class_names import dataset_classes
import argparse
import numpy as np

from od.data.transforms import build_transforms
from od.modeling.detector import build_detection_model
from od.utils import mkdir
from od.utils.checkpoint import CheckPointer
from od.default_config.import_cfg import *


@torch.no_grad()
def run_demo(cfg, ckpt, score_threshold, images_dir, output_dir, dataset_type):
    if dataset_type == "voc":
        class_names = dataset_classes['voc']
    elif dataset_type == 'coco':
        class_names = dataset_classes['coco']
    elif dataset_type == 'kaist':
        class_names = dataset_classes['kaist']
    else:
        raise NotImplementedError('Not implemented now.')

    if not cfg.DATA_LOADER.INCLUDE_BACKGROUND:
        list_class_names = list(class_names)
        list_class_names.remove('__background__')
        class_names = tuple(list_class_names)

    device = torch.device(cfg.MODEL.DEVICE)

    model = build_detection_model(cfg)
    model = model.to(device)
    checkpointer = CheckPointer(model, save_dir=cfg.OUTPUT_DIR)
    checkpointer.load(ckpt, use_latest=ckpt is None)
    weight_file = ckpt if ckpt else checkpointer.get_checkpoint_file()
    print('Loaded weights from {}'.format(weight_file))

    # image_paths = []
    # for root, dirs, files in os.walk(images_dir):
    #     p = [os.path.join(root,dir) for dir in dirs]
    #     p = [os.path.join(p1, 'visible') for p1 in p]
    #
    #     for p1 in p:
    #         image_paths.append(glob.glob(os.path.join(p1, '*.png')))

    image_paths = glob.glob(os.path.join(images_dir, '*.jpg'))

    mkdir(output_dir)

    cpu_device = torch.device("cpu")
    transforms = build_transforms(cfg, is_train=False)
    model.eval()

    # for path in image_paths:
    #
    #     if len(path) == 0:
    #         continue
    #     normalized_path = os.path.normpath(path[0])
    #     path_components = normalized_path.split(os.sep)
    #     new_output_dir = os.path.join(output_dir, path_components[7], path_components[8])
    #     if not os.path.isdir(new_output_dir):
    #         os.mkdir(new_output_dir)

    for i, image_path in enumerate(image_paths):

        if i > 500:
            break
        # if i == 0:
        #     normalized_path = os.path.normpath(image_path)
        #     path_components = normalized_path.split(os.sep)
        #     output_dir = os.path.join(output_dir, path_components[8])
        #     if not os.path.isdir(output_dir):
        #         os.mkdir(output_dir)
        start = time.time()
        image_name = os.path.basename(image_path)

        image = np.array(Image.open(image_path).convert("RGB"))
        height, width = image.shape[:2]
        images = transforms(image)[0].unsqueeze(0)
        load_time = time.time() - start

        start = time.time()
        result = model(images.to(device))[0]
        inference_time = time.time() - start

        result = result.resize((width, height)).to(cpu_device).numpy()
        boxes, labels, scores = result['boxes'], result['labels'], result['scores']

        indices = scores > score_threshold
        boxes = boxes[indices]
        labels = labels[indices].astype(int)
        labels = [label - 1 for label in labels]
        scores = scores[indices]
        meters = ' | '.join(
            [
                'objects {:02d}'.format(len(boxes)),
                'load {:03d}ms'.format(round(load_time * 1000)),
                'inference {:03d}ms'.format(round(inference_time * 1000)),
                'FPS {}'.format(round(1.0 / inference_time))
            ]

        )
        print('({:04d}/{:04d}) {}: {}'.format(i + 1, len(image_paths), image_name, meters))

        drawn_image = draw_boxes(image, boxes, labels, None, class_names).astype(np.uint8)
        Image.fromarray(drawn_image).save(os.path.join(output_dir, image_name))


def main():
    parser = argparse.ArgumentParser(description="SSD Demo.")
    parser.add_argument(
        "--config-file",
        default="",
        metavar="FILE",
        help="path to config file",
        type=str,
    )
    parser.add_argument("--ckpt", type=str, default=None, help="Trained weights.")
    parser.add_argument("--score_threshold", type=float, default=0.7)
    parser.add_argument("--images_dir", default='demo', type=str, help='Specify a image dir to do prediction.')
    parser.add_argument("--output_dir", default='demo/result', type=str, help='Specify a image dir to save predicted images.')
    parser.add_argument("--dataset_type", default="voc", type=str, help='Specify dataset type. Currently support voc and coco.')

    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )
    args = parser.parse_args()
    print(args)

    head = yaml.load(open(args.config_file), Loader=yaml.FullLoader)["MODEL"]["HEAD"]["NAME"]
    sub_cfg = sub_cfg_dict[head]
    cfg.merge_from_other_cfg(sub_cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    print("Loaded configuration file {}".format(args.config_file))
    with open(args.config_file, "r") as cf:
        config_str = "\n" + cf.read()
        print(config_str)
    print("Running with config:\n{}".format(cfg))

    run_demo(cfg=cfg,
             ckpt=args.ckpt,
             score_threshold=args.score_threshold,
             images_dir=args.images_dir,
             output_dir=args.output_dir,
             dataset_type=args.dataset_type)


if __name__ == '__main__':
    main()
