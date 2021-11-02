import argparse
import logging
import yaml
import copy

import torch
import torch.distributed as dist
from numpy import random
from od.engine.inference import do_evaluation
from od.data.build import make_data_loader
from od.engine.trainer import do_train
from od.modeling.detector import build_detection_model
from od.solver import make_optimizer
from od.solver.lr_scheduler import make_lr_scheduler
from od.utils import dist_util, mkdir
from od.utils.checkpoint import CheckPointer
from od.utils.dist_util import synchronize
from od.utils.logger import setup_logger
from od.utils.misc import str2bool
from od.default_config.import_cfg import *
from od.solver.NativeScaler import NativeScaler

def train(cfg, teacher_cfg, args):
    logger = logging.getLogger(cfg.LOGGER.NAME+".trainer")
    model = build_detection_model(cfg)
    teacher_model = build_detection_model(teacher_cfg)

    if args.distributed:
        torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        device = torch.device('cuda:{}'.format(args.local_rank))
        model.to(device)
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True)

        if cfg.KD.ENABLE:
            torch.nn.SyncBatchNorm.convert_sync_batchnorm(teacher_model)
            teacher_model.to(device)
            teacher_model = torch.nn.parallel.DistributedDataParallel(
                teacher_model, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True)
    else:
        device = torch.device(cfg.MODEL.DEVICE)
        model.to(device)
        teacher_model.to(device)

    if cfg.SCHEDULER.TYPE=="Transformer":
        linear_scaled_lr = cfg.SOLVER.LR * cfg.SOLVER.BATCH_SIZE / 512.0
        st_optimizer = make_optimizer(cfg, model, linear_scaled_lr)
        tch_optimizer = make_optimizer(teacher_cfg, teacher_model, linear_scaled_lr)
        Scaler=NativeScaler()
    else:
        lr = cfg.SOLVER.LR * args.num_gpus  # scale by num gpus
        st_optimizer = make_optimizer(cfg, model, lr)
        tch_optimizer = make_optimizer(teacher_cfg, teacher_model, lr)
        Scaler=None

    milestones = [step // args.num_gpus for step in cfg.SOLVER.LR_STEPS]
    tch_scheduler = make_lr_scheduler(teacher_cfg, tch_optimizer, milestones)
    st_scheduler = make_lr_scheduler(cfg, st_optimizer, milestones)

    tch_arguments = {"iteration": 0,"epoch":-1}
    st_arguments = {"iteration": 0,"epoch":-1}

    save_to_disk = dist_util.get_rank() == 0

    if args.pretrained_path != "":
        logger.info("pretrained model found.")
        from torch.nn.parallel import DistributedDataParallel
        checkpoint =torch.load(args.pretrained_path, map_location=torch.device("cpu"))
        if isinstance(model, DistributedDataParallel):
            model_load = model.module
        else:
            model_load = model
        model_load.load_state_dict(checkpoint.pop("model"), strict=False)

    tch_checkpointer = CheckPointer(teacher_model, tch_optimizer, tch_scheduler, teacher_cfg.OUTPUT_DIR, save_to_disk, logger)
    tch_extra_checkpoint_data = tch_checkpointer.load()
    tch_arguments.update(tch_extra_checkpoint_data)

    st_checkpointer = CheckPointer(model, st_optimizer, st_scheduler, cfg.OUTPUT_DIR, save_to_disk, logger)
    st_extra_checkpoint_data = st_checkpointer.load()
    st_arguments.update(st_extra_checkpoint_data)

    if cfg.SOLVER.MAX_ITER==None:
        train_loader = make_data_loader(teacher_cfg, cfg, is_train=True, distributed=args.distributed)
    else:
        max_iter = cfg.SOLVER.MAX_ITER // args.num_gpus
        train_loader = make_data_loader(teacher_cfg, cfg, is_train=True, distributed=args.distributed, max_iter=max_iter,
                                      start_iter=tch_arguments['iteration'])

    tch_model, st_model = do_train(cfg, teacher_cfg, model, teacher_model, train_loader, tch_optimizer, tch_scheduler, st_optimizer, st_scheduler,
                                   tch_checkpointer, st_checkpointer, device, tch_arguments, st_arguments, args, scaler=Scaler)

    return model


def main():
    #arguments
    #any New config should be added  to config file and you  pass this config file at the arguments
    parser = argparse.ArgumentParser(description='Single Shot MultiBox Detector Training With PyTorch')
    parser.add_argument("--config-file", default="", metavar="FILE", required=True, help="path to student config file (Thermal)",type=str,)
    parser.add_argument("--teacher-config-file", default="", metavar="FILE",help="path to teacher config file (RGB)",type=str,)
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument('--log_step', default=10, type=int, help='Print logs every log_step')
    parser.add_argument('--save_step', default=2500, type=int, help='Save checkpoint every save_step')
    parser.add_argument('--save_epoch', default=5, type=int, help='Save checkpoint every epoch_step')
    parser.add_argument('--eval_step', default=2500, type=int, help='Evaluate dataset every eval_step, disabled when eval_step < 0')
    parser.add_argument('--pretrained_path', default="", type=str, help='pretrianed path , default is empty, if you put any path it will pretrain the model')
    parser.add_argument('--use_tensorboard', default=True, type=str2bool)
    parser.add_argument('--seed',default=0,type=int,help='Random seed for processes. Seed must be fixed for distributed training')
    parser.add_argument("--skip-test",dest="skip_test",help="Do not test the final model",action="store_true",)
    parser.add_argument("--calc_energy", action='store_true',help='If set, measures and outputs the Energy consumption')
    parser.add_argument("opts",help="Modify config options using the command-line",default=None,nargs=argparse.REMAINDER,)
    parser.add_argument("--precision_display", action="store_true", help="If set, display precision and recall")
    parser.add_argument("--lamr", action="store_true", help="If set, display LAMR result")
    parser.add_argument("--iou_threshold", type=float, default=0.5, help="The default IOU threshold for precision display")
    parser.add_argument("--dataset_type", default="", type=str, help="Specify dataset type. FLIR or KAIST.")
    args = parser.parse_args()

    num_gpus = torch.cuda.device_count()
    args.distributed = num_gpus > 1
    args.num_gpus = num_gpus
    # remove torch.backends.cudnn.benchmark to avoid potential risk
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    if args.distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(
            backend="nccl", init_method="env://", world_size=num_gpus, rank=args.local_rank)
        synchronize()

    # defined by 'head' not meta_architecture
    head = yaml.load(open(args.config_file), Loader=yaml.FullLoader)["MODEL"]["HEAD"]["NAME"]
    sub_cfg = sub_cfg_dict[head]

    cfg.merge_from_other_cfg(sub_cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    #if cfg.KD.ENABLE:
    teacher_cfg = copy.deepcopy(cfg)
    if not args.teacher_config_file:
        raise RuntimeError("Requires teacher config file when enable_KD is set")

    teacher_head = yaml.load(open(args.teacher_config_file), Loader=yaml.FullLoader)["MODEL"]["HEAD"]["NAME"]
    sub_cfg = sub_cfg_dict[teacher_head]
    teacher_cfg.merge_from_other_cfg(sub_cfg)
    teacher_cfg.merge_from_file(args.teacher_config_file)
    # else:
    #     teacher_cfg = None

    if cfg.OUTPUT_DIR:
        mkdir(cfg.OUTPUT_DIR)

    logger = setup_logger(cfg.LOGGER.NAME, dist_util.get_rank(), cfg.OUTPUT_DIR)
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

    tch_model, st_model = train(cfg, teacher_cfg, args)

    # if not args.skip_test:
    #     logger.info('Start evaluating...')
    #     torch.cuda.empty_cache()  # speed up evaluating after training finished
    #     do_evaluation(cfg, model, distributed=args.distributed,
    #                   precision_display=args.precision_display,
    #                   iou_threshold=args.iou_threshold,
    #                   dataset_type=args.dataset_type)


if __name__ == '__main__':
    main()
