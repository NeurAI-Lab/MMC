#!/usr/bin/env python3
import argparse
import logging
import os
import glob
import yaml

import csv
import numpy as np
import torch
import torch.utils.data
from os import path

from od.engine.inference import do_evaluation
from od.modeling.detector import build_detection_model
from od.utils import dist_util, mkdir
from od.utils.checkpoint import CheckPointer
from od.utils.dist_util import synchronize
from od.utils.logger import setup_logger
from od.utils.flops_counter import get_model_complexity_info
from od.utils.energy_meter import EnergyMeter
from contextlib import ExitStack
from od.default_config.import_cfg import *

def results_csv(append_csv, csv_dir):

    if not append_csv:
        if csv_dir == 'None':
            file = os.path.join(cfg.OUTPUT_DIR, "results.csv")
        else:
            file = os.path.join(csv_dir, "results.csv")
        if path.exists(file):
            os.remove(file)
    else:
        file = os.path.join(csv_dir, "results.csv")
    return file

def evaluation(cfg, args, distributed):
    logger = logging.getLogger("Object Detection.inference")

    ckpt = args.ckpt
    eval_only = args.eval_only
    calc_energy = args.calc_energy
    lamr = args.lamr
    precision_display = args.precision_display
    iou_threshold = args.iou_threshold
    dataset_type = args.dataset_type
    args_outputdir = args.output_dir
    write_csv = args.write_csv
    csv_dir = args.results_csv_dir
    append_csv = args.append_csv

    model = build_detection_model(cfg)

    checkpointer = CheckPointer(model, save_dir=cfg.OUTPUT_DIR, logger=logger)
    device = torch.device(cfg.MODEL.DEVICE)
    model.to(device)
    checkpointer.load(ckpt, use_latest=ckpt is None)

    if not eval_only:
        image_res = (3, cfg.INPUT.IMAGE_SIZE, cfg.INPUT.IMAGE_SIZE)
        flops_count, params_count = get_model_complexity_info(model, image_res)
        print("MAC Count:", flops_count)
        print("Number of Parameters:", params_count)

    if not cfg.OUTPUT_DIR:
        if args_outputdir:
            cfg.defrost()
            cfg.OUTPUT_DIR = args_outputdir

    energy_mtr = EnergyMeter(dir=cfg.OUTPUT_DIR, dataset=cfg.DATASETS.TEST[0])
    with EnergyMeter(dir=cfg.OUTPUT_DIR) if calc_energy else ExitStack():
        inf_results, eval_results = do_evaluation(cfg, model, distributed, get_inf_time=False if eval_only else True,
                                                  lamr = lamr,
                                                  precision_display=precision_display,
                                                  iou_threshold=iou_threshold,
                                                  dataset_type=dataset_type)

    if write_csv:
        file = results_csv(append_csv, csv_dir)

    if not eval_only:
        for i, dataset_name in enumerate(cfg.DATASETS.TEST):
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference", dataset_name)
            list_of_files = glob.glob(output_folder + "**/*.txt")  # * means all if need specific format then *.csv
            result_path = max(list_of_files, key=os.path.getmtime)

            result_strings = []
            result_strings.append("\nInference Speed (in FPS): {:.4f}".format(1000 / inf_results[i]))
            result_strings.append("MAC Count (in GMac): {}".format(flops_count))
            result_strings.append("Number of Parameters (in M): {}".format(params_count))
            with open(result_path, "a") as f:
                f.write('\n'.join(result_strings))

            if write_csv:
                names = ['Head', 'Backbone', 'data_ann', 'data_dir', 'data_path', 'mAP', 'mAP50', 'mAP75', 'mAPs', 'mAPm', 'mAPl', "AP{}".format(iou_threshold),
                         "AR{}".format(iou_threshold), 'mAR', 'LAMR', 'F1-score', 'GMac',
                         'Num_Params', 'Energy', 'Inf_speed', 'model_output_dir', 'config_path']

                if dataset_type == 'voc':
                    eval_results[i]['metrics']['AP'] = eval_results[i]['metrics']['mAP']
                    eval_results[i]['metrics']["AP{}".format(iou_threshold)] = -1
                    eval_results[i]['metrics']["AR{}".format(iou_threshold)] = -1
                    eval_results[i]['metrics']['AP50'] = -0
                    eval_results[i]['metrics']['AP75'] = -0
                    eval_results[i]['metrics']['APs'] = -0
                    eval_results[i]['metrics']['APm'] = -0
                    eval_results[i]['metrics']['APl'] = -0

                values = [cfg.MODEL.HEAD.NAME, cfg.MODEL.BACKBONE.NAME, dataset_name, cfg.DATASETS.DATA_DIR, cfg.DATASETS.PATH, round(eval_results[i]['metrics']['AP'],5)*100,
                          round(eval_results[i]['metrics']['AP50'],5)*100, round(eval_results[i]['metrics']['AP75'],5)*100,
                          round(eval_results[i]['metrics']['APs'],5)*100, round(eval_results[i]['metrics']['APm'],5)*100,
                          round(eval_results[i]['metrics']['APl'],5)*100, round(eval_results[i]['metrics']["AP{}".format(iou_threshold)],5)*100,
                          round(eval_results[i]['metrics']["AR{}".format(iou_threshold)],5)*100, round(eval_results[i]['metrics']["Avg Recall"],5)*100,
                          round(eval_results[i]['metrics']["LAMR"],5)*100,
                          round(eval_results[i]['metrics']["F1-score"],5)*100,
                          flops_count, params_count,
                          energy_mtr.energy if calc_energy else 0, round((1000 / inf_results[i]),4), cfg.OUTPUT_DIR, args.config_file]

                if not append_csv:
                    np.savetxt(file, (names,values), delimiter=',', fmt='%s')
                    append_csv = True
                else:
                    with open(file, 'a') as f:
                        writer = csv.writer(f)
                        writer.writerow(values)

def main():
    parser = argparse.ArgumentParser(description='Evaluation on VOC and COCO dataset.')
    parser.add_argument(
        "--config-file",
        default="",
        metavar="FILE",
        help="path to config file",
        type=str,
        required=True
    )
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument(
        "--ckpt",
        help="The path to the checkpoint for test, default is the latest checkpoint.",
        default=None,
        type=str,
    )

    parser.add_argument("--output_dir", default="eval_results", type=str, help="The directory to store evaluation results.")

    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )
    parser.add_argument("--eval_only", action='store_true',
                        help='If set, outputs MAP without other statistics such as Inference time, Energy and Number of Parameters')
    parser.add_argument("--calc_energy", action='store_true',
                        help='If set, measures and outputs the Energy consumption')
    parser.add_argument("--precision_display", action="store_true", help="If set, display precision, recall and F-score")
    parser.add_argument("--lamr", action="store_true", help="If set, display LAMR result")
    parser.add_argument("--iou_threshold", type=float, default=0.5, help="The default IOU threshold for precision display")
    parser.add_argument("--dataset_type", default="", type=str, help="Specify dataset type. Currently support voc and coco.")
    parser.add_argument("--write_csv", action='store_true', help="Write results to csv file")
    parser.add_argument("--results_csv_dir", default="None", type=str, help="specify a directory for results csv")
    parser.add_argument("--append_csv", action='store_true', help="If append to existing csv")

    args = parser.parse_args()

    num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    distributed = num_gpus > 1

    if torch.cuda.is_available():
        # This flag allows you to enable the inbuilt cudnn auto-tuner to
        # find the best algorithm to use for your hardware.
        torch.backends.cudnn.benchmark = True
    if distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend="nccl", init_method="env://")
        synchronize()

    head = yaml.load(open(args.config_file), Loader=yaml.FullLoader)["MODEL"]["HEAD"]["NAME"]
    sub_cfg = sub_cfg_dict[head]

    cfg.merge_from_other_cfg(sub_cfg)
    cfg.merge_from_file(args.config_file)

    # if args.opts:
    #     pair = args.opts[1].split("-")
    #     args.opts = [args.opts[0], (pair[0], pair[1], pair[2], pair[3])]
    #so we can use opts to specify dataset tuples without modifying the config file all the time
    # for example,
    # python test.py --config-file <> --ckpt <> DATASETS.TEST "coco_new_val"
    if args.opts:
        if args.opts[0] == 'DATASETS.TEST' or args.opts[0] == 'DATASETS.TRAIN':
            pair = args.opts[1].split("-")
            if len(pair) == 1:
                name = (pair[0], )
            else:
                name = (pair[i] for i in range(len(pair)))
            args.opts[1] = name
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    if not cfg.OUTPUT_DIR or args.output_dir:
        cfg.defrost()
        cfg.OUTPUT_DIR = args.output_dir
        if not os.path.exists(cfg.OUTPUT_DIR):
            mkdir(cfg.OUTPUT_DIR)

    logger = setup_logger("Object Detection", dist_util.get_rank(), cfg.OUTPUT_DIR)
    logger.info("Using {} GPUs".format(num_gpus))
    logger.info(args)

    if args.precision_display and args.dataset_type == "":
        logger.info('Precision display argument requires dataset type')
        return

    logger.info("Loaded configuration file {}".format(args.config_file))
    with open(args.config_file, "r") as cf:
        config_str = "\n" + cf.read()
        logger.info(config_str)
    logger.info("Running with config:\n{}".format(cfg))
    evaluation(cfg, args,
               distributed=distributed)


if __name__ == '__main__':
    main()
