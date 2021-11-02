import collections
import datetime
import logging
import os
import time
import torch
import numpy as np
import torch.distributed as dist

from od.engine.inference import do_evaluation
from od.utils import dist_util
from od.utils.metric_logger import MetricLogger
from od.utils.energy_meter import EnergyMeter
from contextlib import ExitStack
from od.modeling.detector.distillation import KD_detector, collate_loss
from od.modeling.detector.head_loss import head_loss, ssd_loss
from od.modeling.decoder.loss import recon_loss
from collections import Counter


def write_metric(eval_result, prefix, summary_writer, global_step):
    for key in eval_result:
        value = eval_result[key]
        tag = '{}/{}'.format(prefix, key)
        if isinstance(value, collections.Mapping):
            write_metric(value, tag, summary_writer, global_step)
        else:
            summary_writer.add_scalar(tag, value, global_step=global_step)


def reduce_loss_dict(loss_dict):
    """
    Reduce the loss dictionary from all processes so that process with rank
    0 has the averaged results. Returns a dict with the same fields as
    loss_dict, after reduction.
    """
    world_size = dist_util.get_world_size()
    if world_size < 2:
        return loss_dict
    with torch.no_grad():
        loss_names = []
        all_losses = []
        for k in sorted(loss_dict.keys()):
            loss_names.append(k)
            all_losses.append(loss_dict[k])
        all_losses = torch.stack(all_losses, dim=0)
        dist.reduce(all_losses, dst=0)
        if dist.get_rank() == 0:
            # only main process gets accumulated, so only divide by
            # world_size in this case
            all_losses /= world_size
        reduced_losses = {k: v for k, v in zip(loss_names, all_losses)}
    return reduced_losses

def model_train_core(st_cfg, tch_cfg, st_model, tch_model,data_loader,
             tch_optimizer, st_optimizer, images_rgb, images_thm, targets_rgb, targets_thm, meters):

    #Method - swap features between RGB and Thermal
    if tch_cfg.KD.SWAP_FEATURES:
        features_bt = tch_model(images_rgb, targets=targets_rgb)
        features_bs = st_model(images_thm, targets=targets_thm)
        for layer in tch_cfg.KD.SWAP_LAYERS:
            tmp = features_bt[layer]
            features_bt[layer] = features_bs[layer]
            features_bs[layer] = tmp
        features_bt, features_ht = tch_model(images_rgb, targets=targets_rgb, modifiedfts=features_bt)
        features_bs, features_hs = st_model(images_thm, targets=targets_thm, modifiedfts=features_bs)
    # Method - Feature Fusion of features between RGB and Thermal
    elif tch_cfg.KD.CONCAT_FEATURES:
        features_bt = tch_model(images_rgb, targets=targets_rgb)
        features_bs = st_model(images_thm, targets=targets_thm)
        for layer in tch_cfg.KD.CONCAT_LAYERS:
            concat_ft = torch.cat((features_bt[layer], features_bs[layer]), 1)
            features_bt[layer] = concat_ft
            features_bs[layer] = concat_ft
        features_bt, features_ht = tch_model(images_rgb, targets=targets_rgb, modifiedfts=features_bt)
        features_bs, features_hs = st_model(images_thm, targets=targets_thm, modifiedfts=features_bs)
    # Method - Input Fusion of features between RGB and Thermal
    elif tch_cfg.KD.CONCAT_INPUT:
        con_images = torch.cat((images_rgb, images_thm), 1)
        features_ht = tch_model(con_images, targets=targets_rgb)
    # Method - Reconstruction as Auxiliary task
    elif tch_cfg.KD.AUX_RECON:
        features_bt, features_ht, features_recon_t = tch_model(images_rgb, targets=targets_rgb)
        features_bs, features_hs, features_recon_s = st_model(images_thm, targets=targets_thm)
    else:
        features_bt, features_ht = tch_model(images_rgb, targets=targets_rgb)
        features_bs, features_hs = st_model(images_thm, targets=targets_thm)

    if tch_cfg.KD.ENABLE_DML == True:
        if tch_cfg.MODEL.HEAD.NAME == 'CenterNetHead':
            kd_loss_tch = {}
            kd_loss_tch['loss'] = 0
            KD_detector(tch_cfg, features_bt, features_bs, features_ht['hm'], features_hs['hm'].detach(),
                        targets_thm, kd_loss_tch)

            kd_loss_st = {}
            kd_loss_st['loss'] = 0
            KD_detector(tch_cfg, features_bt, features_bs, features_hs['hm'], features_ht['hm'].detach(),
                        targets_thm, kd_loss_st)

            tch_loss = head_loss(tch_cfg, features_ht, targets=targets_rgb)
            st_loss = head_loss(st_cfg, features_hs, targets=targets_thm)
            loss_dict_tch = dict(Counter(kd_loss_tch) + Counter(tch_loss))
            loss_dict_st = dict(Counter(kd_loss_st) + Counter(st_loss))

        if tch_cfg.MODEL.HEAD.NAME == 'SSDBoxHead':
            kd_loss_tch = {}
            kd_loss_tch['loss'] = 0
            KD_detector(tch_cfg, features_bt, [fs.detach() for fs in features_bs], features_ht[0],
                        features_hs[0].detach(), targets_thm, kd_loss_tch)

            kd_loss_st = {}
            kd_loss_st['loss'] = 0
            KD_detector(st_cfg, features_bs, [ft.detach() for ft in features_bt], features_hs[0],
                        features_ht[0].detach(), targets_thm, kd_loss_st)

            tch_loss = ssd_loss(tch_cfg, features_ht, targets=targets_rgb)
            st_loss = ssd_loss(st_cfg, features_hs, targets=targets_thm)

            if tch_cfg.KD.AUX_RECON:

                if isinstance(tch_model, torch.nn.parallel.DistributedDataParallel):
                    tch_model = tch_model.module
                recon_img_t, recon_img_s = tch_model.recon(features_recon_t, features_recon_s)
                if tch_cfg.KD.AUX_RECON_MODE == 'normal':
                    recon_loss_tch = recon_loss(images_rgb, recon_img_t)
                    recon_loss_st = 0
                elif tch_cfg.KD.AUX_RECON_MODE == 'cross':
                    recon_loss_tch = recon_loss(images_rgb, recon_img_t)
                    recon_loss_st = recon_loss(images_rgb, recon_img_s)
                # # Plot Reconstructed Image
                # b = recon_img_t * 255
                # b = b[0, :]
                # b = b.permute(1, 2, 0)
                # c = b.cpu().detach().numpy().astype(np.uint8)
                # from PIL import Image
                # pil_img = Image.fromarray(c, "RGB")
                # pil_img.save('dl/y.jpg')
                #
                # b = recon_img_s * 255
                # b = b[0, :]
                # b = b.permute(1, 2, 0)
                # c = b.cpu().detach().numpy().astype(np.uint8)
                # from PIL import Image
                # pil_img = Image.fromarray(c, "RGB")
                # pil_img.save('dl/y1.jpg')
                # # plot
                # b = images_rgb * 255
                # b = b[0, :]
                # b = b.permute(1, 2, 0)
                # c = b.cpu().detach().numpy().astype(np.uint8)
                # from PIL import Image
                # pil_img = Image.fromarray(c, "RGB")
                # pil_img.save('dl/z.jpg')
            loss_dict_tch = {}
            loss_dict_st = {}
            collate_loss(tch_cfg, tch_loss, kd_loss_tch, loss_dict_tch, True)
            collate_loss(st_cfg, st_loss, kd_loss_st, loss_dict_st, False)

            if tch_cfg.KD.AUX_RECON:
                loss_dict_tch['tch_recon_loss'] = recon_loss_tch
                loss_dict_tch['st_recon_loss'] = recon_loss_st
                loss_dict_tch['loss'] += tch_cfg.KD.LOSS_WEIGHTS['l2_recon'] * recon_loss_tch
                loss_dict_st['loss'] += st_cfg.KD.LOSS_WEIGHTS['l2_recon'] * recon_loss_st
    elif tch_cfg.KD.ENABLE == False:
        tch_loss = {}
        tch_loss = ssd_loss(tch_cfg, features_ht, targets=targets_rgb)
        loss_dict_tch = {}
        collate_loss(tch_cfg, tch_loss, 0, loss_dict_tch, True)
    else:
        tch_loss = {}
        tch_loss = ssd_loss(tch_cfg, features_ht, targets=targets_rgb)
        st_loss = {}
        st_loss = ssd_loss(st_cfg, features_hs, targets=targets_thm)
        loss_dict_tch = {}
        loss_dict_st = {}
        collate_loss(tch_cfg, tch_loss, 0, loss_dict_tch, True)
        collate_loss(st_cfg, st_loss, 0, loss_dict_st, False)

    loss_tch = loss_dict_tch['loss']
    loss_dict_reduced_tch = reduce_loss_dict(loss_dict_tch)
    losses_reduced_tch = loss_dict_reduced_tch['loss']
    loss_dict_reduced_st = {}
    meters.update(teacher_loss=losses_reduced_tch, **loss_dict_reduced_tch)

    if tch_cfg.KD.ENABLE == True:
        loss_st = loss_dict_st['loss']
        loss_dict_reduced_st = reduce_loss_dict(loss_dict_st)
        losses_reduced_st = loss_dict_reduced_st['loss']
        meters.update(student_loss=losses_reduced_st, **loss_dict_reduced_st)

    tch_optimizer.zero_grad()
    if tch_cfg.KD.ENABLE == True:
        st_optimizer.zero_grad()

    loss_tch.backward(retain_graph=True)
    if tch_cfg.KD.ENABLE == True:
        loss_st.backward(retain_graph=True)

    tch_optimizer.step()
    if tch_cfg.KD.ENABLE == True:
        st_optimizer.step()

    return loss_dict_reduced_st, loss_dict_reduced_tch

def do_train_iteration(st_cfg, tch_cfg, st_model, tch_model,data_loader,
             tch_optimizer, st_optimizer, tch_scheduler,st_scheduler, tch_checkpointer, st_checkpointer,
             device, tch_arguments, st_arguments, start_iter, summary_writer, meters, max_iter, logger, endTime, args,scaler=None,):

    for iteration, (teacher, student) in enumerate(data_loader, start_iter):
        iteration = iteration + 1
        tch_arguments["iteration"] = iteration
        st_arguments["iteration"] = iteration

        images_rgb, targets_rgb, _ = teacher
        images_thm, targets_thm, _ = student
        images_rgb = images_rgb.to(device)
        images_thm = images_thm.to(device)
        targets_rgb = targets_rgb.to(device)
        targets_thm = targets_thm.to(device)

        loss_dict_reduced_st, loss_dict_reduced_tch = model_train_core(st_cfg, tch_cfg, st_model, tch_model, data_loader, tch_optimizer, st_optimizer, images_rgb, images_thm,
                                                                       targets_rgb, targets_thm, meters)

        tch_scheduler.step()
        if tch_cfg.KD.ENABLE == True:
            st_scheduler.step()

        batch_time = time.time() - endTime
        end = time.time()
        meters.update(time=batch_time)
        if iteration % args.log_step == 0:
            eta_seconds = meters.time.global_avg * (max_iter - iteration)
            eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
            logger.info(
                meters.delimiter.join([
                    "iter: {iter:06d}",
                    "epo: {epo:d}",
                    "lr: {lr:.9f}",
                    "ImShape: {ImShape[0]}, {ImShape[1]}",
                    '{meters}',
                    "eta: {eta}",
                    'mem: {mem}M',
                ]).format(
                    iter=iteration,
                    epo=iteration // (len(data_loader.dataset) // tch_cfg.SOLVER.BATCH_SIZE),
                    lr=tch_optimizer.param_groups[0]['lr'],
                    ImShape=(images_rgb.data.size(2), images_rgb.data.size(3)),
                    meters=str(meters),
                    eta=eta_string,
                    mem=round(torch.cuda.max_memory_allocated() / 1024.0 / 1024.0),
                )
            )
            if summary_writer:
                global_step = iteration

                for loss_name, loss_item in loss_dict_reduced_st.items():
                    summary_writer.add_scalar('Student losses/{}'.format(loss_name), loss_item, global_step=global_step)
                for loss_name, loss_item in loss_dict_reduced_tch.items():
                    summary_writer.add_scalar('Teacher losses/{}'.format(loss_name), loss_item, global_step=global_step)

                summary_writer.add_scalar('lr', tch_optimizer.param_groups[0]['lr'], global_step=global_step)

        if iteration % args.save_step == 0:
            tch_checkpointer.save("model_{:06d}".format(iteration), **tch_arguments)
            if tch_cfg.KD.ENABLE == True:
                st_checkpointer.save("model_{:06d}".format(iteration), **st_arguments)

        if args.eval_step > 0 and iteration % args.eval_step == 0 and not iteration == max_iter:
            inf_time, tch_eval_results = do_evaluation(tch_cfg, tch_model, distributed=args.distributed,
                                                       iteration=iteration, precision_display=args.precision_display,
                                                       iou_threshold=args.iou_threshold, dataset_type=args.dataset_type)
            if dist_util.get_rank() == 0 and summary_writer:
                for eval_result, dataset in zip(tch_eval_results, tch_cfg.DATASETS.TEST):
                    write_metric(eval_result['metrics'], 'tch_metrics/' + dataset, summary_writer, iteration)
            tch_model.train()  # *IMPORTANT*: change to train mode after eval.

            if tch_cfg.KD.ENABLE == True:
                inf_time, st_eval_results = do_evaluation(st_cfg, st_model, distributed=args.distributed,
                                                          iteration=iteration, precision_display=args.precision_display,
                                                           iou_threshold=args.iou_threshold, dataset_type=args.dataset_type)
                if dist_util.get_rank() == 0 and summary_writer:
                    for eval_result, dataset in zip(st_eval_results, st_cfg.DATASETS.TEST):
                        write_metric(eval_result['metrics'], 'st_metrics/' + dataset, summary_writer, iteration)
                st_model.train()  # *IMPORTANT*: change to train mode after eval.


def do_train_epochs(st_cfg, tch_cfg, st_model, tch_model,data_loader,
             tch_optimizer, st_optimizer, tch_scheduler,st_scheduler, tch_checkpointer, st_checkpointer,
             device, tch_arguments, st_arguments, start_iter, summary_writer, meters, max_iter, logger, endTime, args,scaler=None,):

    iteration=start_iter
    start_epoch=tch_arguments["epoch"]+1

    for epoch_iter in range(start_epoch,tch_cfg.SOLVER.MAX_EPOCHS):

        for _, (teacher, student) in enumerate(data_loader):
            iteration = iteration + 1

            images_rgb, targets_rgb, _ = teacher
            images_thm, targets_thm, _ = student
            images_rgb = images_rgb.to(device)
            images_thm = images_thm.to(device)
            targets_rgb = targets_rgb.to(device)
            targets_thm = targets_thm.to(device)
            loss_dict_reduced_st, loss_dict_reduced_tch = model_train_core(st_cfg, tch_cfg, st_model, tch_model, data_loader, tch_optimizer, st_optimizer, images_rgb, images_thm,
                                                                           targets_rgb, targets_thm, meters)

            batch_time = time.time() - endTime
            endTime = time.time()
            meters.update(time=batch_time)
            if iteration % args.log_step == 0:
                # eta_seconds = meters.time.global_avg * (max_iter - iteration)
                # eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                logger.info(
                    meters.delimiter.join([
                        "iter: {iter:06d}",
                        "epo: {epo:d}",
                        "lr: {lr:.9f}",
                        "ImShape: {ImShape[0]}, {ImShape[1]}",
                        '{meters}',
                        'mem: {mem}M',
                    ]).format(
                        iter=iteration,
                        epo=epoch_iter,
                        lr=tch_optimizer.param_groups[0]['lr'],
                        ImShape=(images_rgb.data.size(2), images_rgb.data.size(3)),
                        meters=str(meters),
                        mem=round(torch.cuda.max_memory_allocated() / 1024.0 / 1024.0),
                    )
                )
                if summary_writer:
                    global_step = iteration

                    for loss_name, loss_item in loss_dict_reduced_st.items():
                        summary_writer.add_scalar('Student losses/{}'.format(loss_name), loss_item,
                                                  global_step=global_step)
                    for loss_name, loss_item in loss_dict_reduced_tch.items():
                        summary_writer.add_scalar('Teacher losses/{}'.format(loss_name), loss_item,
                                                  global_step=global_step)
                    summary_writer.add_scalar('lr', tch_optimizer.param_groups[0]['lr'], global_step=global_step)

            if args.eval_step > 0 and iteration % args.eval_step == 0 and not iteration == max_iter:
                inf_time, tch_eval_results = do_evaluation(tch_cfg, tch_model, distributed=args.distributed,
                                                           iteration=iteration, lamr=args.lamr, precision_display=args.precision_display,
                                                           iou_threshold=args.iou_threshold, dataset_type=args.dataset_type)
                if dist_util.get_rank() == 0 and summary_writer:
                    for eval_result, dataset in zip(tch_eval_results, tch_cfg.DATASETS.TEST):
                        write_metric(eval_result['metrics'], 'tch_metrics/' + dataset, summary_writer, iteration)
                tch_model.train()  # *IMPORTANT*: change to train mode after eval.

                if tch_cfg.KD.ENABLE == True:
                    inf_time, st_eval_results = do_evaluation(st_cfg, st_model, distributed=args.distributed,
                                                              iteration=iteration, lamr=args.lamr, precision_display=args.precision_display,
                                                              iou_threshold=args.iou_threshold, dataset_type=args.dataset_type)
                    if dist_util.get_rank() == 0 and summary_writer:
                        for eval_result, dataset in zip(st_eval_results, st_cfg.DATASETS.TEST):
                            write_metric(eval_result['metrics'], 'st_metrics/' + dataset, summary_writer, iteration)
                    st_model.train()  # *IMPORTANT*: change to train mode after eval.

        tch_arguments["iteration"] = iteration
        tch_arguments["epoch"] = epoch_iter
        tch_scheduler.step(epoch_iter)
        if epoch_iter % args.save_epoch == 0:
            tch_checkpointer.save("model_{:06d}".format(iteration), **tch_arguments)

        if tch_cfg.KD.ENABLE == True:
            st_arguments["iteration"] = iteration
            st_arguments["epoch"] = epoch_iter
            st_scheduler.step(epoch_iter)
            if epoch_iter % args.save_epoch == 0:
                st_checkpointer.save("model_{:06d}".format(iteration), **st_arguments)

def do_train(st_cfg, tch_cfg, st_model, tch_model,
             data_loader,
             tch_optimizer,
             tch_scheduler,
             st_optimizer,
             st_scheduler,
             tch_checkpointer,
             st_checkpointer,
             device,
             tch_arguments,
             st_arguments,
             args,scaler=None):
    logger = logging.getLogger(tch_cfg.LOGGER.NAME + ".trainer")
    logger.info("Start training ...")
    meters = MetricLogger()

    st_model.train()
    tch_model.train()
    save_to_disk = dist_util.get_rank() == 0
    if args.use_tensorboard and save_to_disk and args.local_rank == 0:
        import tensorboardX

        summary_writer = tensorboardX.SummaryWriter(log_dir=os.path.join(tch_cfg.OUTPUT_DIR, 'tf_logs'))
    else:
        summary_writer = None

    max_iter = len(data_loader)
    start_iter = tch_arguments["iteration"]
    start_training_time = time.time()
    end = time.time()

    with EnergyMeter(writer=summary_writer, dir=tch_checkpointer.save_dir) if args.calc_energy else ExitStack():

        # Epochs training for transformer-based methods
        if tch_cfg.SOLVER.MAX_EPOCHS!= None and tch_cfg.SOLVER.MAX_ITER== None:
            do_train_epochs(st_cfg, tch_cfg, st_model, tch_model,data_loader,tch_optimizer,st_optimizer,tch_scheduler,st_scheduler,tch_checkpointer,st_checkpointer,
                               device,tch_arguments,st_arguments,start_iter,summary_writer,meters,max_iter,logger,end,args, scaler )
        # Iteration training for CNN-based methods
        elif tch_cfg.SOLVER.MAX_EPOCHS== None and tch_cfg.SOLVER.MAX_ITER!= None :
            do_train_iteration(st_cfg, tch_cfg, st_model, tch_model,data_loader,tch_optimizer,st_optimizer,tch_scheduler,st_scheduler,tch_checkpointer,st_checkpointer,
                               device,tch_arguments,st_arguments,start_iter,summary_writer,meters,max_iter,logger,end,args, scaler )
        else:
            assert False,"Wrong Solver"

    tch_checkpointer.save("model_final", **tch_arguments)
    st_checkpointer.save("model_final", **st_arguments)
    # compute training time
    total_training_time = int(time.time() - start_training_time)
    total_time_str = str(datetime.timedelta(seconds=total_training_time))
    logger.info("Total training time: {} ({:.4f} s / it)".format(total_time_str, total_training_time / max_iter))
    return tch_model, st_model
