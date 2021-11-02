import torch
import random
from torch.utils.data import DataLoader
from torch.utils.data._utils.collate import default_collate

from od.data import samplers
from od.data.datasets import build_dataset
from od.data.transforms import build_transforms, build_target_transform
from od.structures.container import Container
from od.utils.yolo_utils import resize, get_different_scale

class TestBatchCollator:
    def __init__(self, is_train=True,head_name=""):
        self.is_train = is_train
        self.head_name=head_name

    def __call__(self, batch):
        transposed_batch = list(zip(*batch))
        images = default_collate(transposed_batch[0])
        img_ids = default_collate(transposed_batch[2])

        if self.is_train:
            list_targets = transposed_batch[1]
            if "FCOS" in self.head_name:
                targets=list_targets
            elif "NanoDet" in self.head_name:
                targets = Container(
                    {key: ([d[key] for d in list_targets]) for key in list_targets[0]}
                )
            else:
                targets = Container(
                    {key: default_collate([d[key] for d in list_targets]) for key in list_targets[0]}
                )
        else:
            targets = None
        return images, targets, img_ids

class BatchCollator:
    def __init__(self, is_train=True,head_name=""):
        self.is_train = is_train
        self.head_name=head_name

    def __call__(self, batch):
        transposed_batch = list(zip(*batch))

        tch_img = []
        img_id = []
        tch_targets = []
        for i in transposed_batch[0]:
            tch_img.append(i[0])
            img_id.append(i[2])
            tch_targets.append(i[1])
        tch_images = default_collate(tch_img)
        img_ids = default_collate(img_id)

        st_img = []
        img_id = []
        st_targets = []
        for i in transposed_batch[1]:
            st_img.append(i[0])
            st_targets.append(i[1])
        st_images = default_collate(st_img)

        if self.is_train:
            list_targets = transposed_batch[1]
            tch_targets = Container(
                {key: default_collate([d[key] for d in tch_targets]) for key in tch_targets[0]}
            )
            st_targets = Container(
                {key: default_collate([d[key] for d in st_targets]) for key in st_targets[0]}
            )
        else:
            tch_targets = None
            st_targets = None
        return (tch_images, tch_targets, img_ids), (st_images, st_targets, img_ids)

class MultiScaleBatchCollator:
    def __init__(self, is_train=True, batch_size=8):
        self.is_train = is_train
        self.batch_count = 0
        # 320 to 608
        self.random_range = (10, 19)
        self.batch_size = batch_size

    def __call__(self, batch):
        transposed_batch = list(zip(*batch))
        images = default_collate(transposed_batch[0])
        img_ids = default_collate(transposed_batch[2])

        if self.is_train and self.batch_count > 0 and self.batch_count % 10 == 0:
            img_size = random.randint(*self.random_range) * 32
            # Resize images to input shape
            images = torch.stack([resize(img, img_size) for img in images])
        self.batch_count += 1

        if self.is_train:
            list_targets = transposed_batch[1]
            targets = Container(
                {key: default_collate([d[key] for d in list_targets]) for key in list_targets[0]}
            )
        else:
            targets = None
        return images, targets, img_ids

def make_data_loader(tch_cfg, st_cfg, is_train=True, distributed=False, max_iter=None, start_iter=0):
    tch_train_transform = build_transforms(tch_cfg, is_train=is_train)
    st_train_transform = build_transforms(st_cfg, is_train=is_train)
    target_transform = build_target_transform(tch_cfg) if is_train else None
    if is_train:
        tch_dataset_list = tch_cfg.DATASETS.TRAIN
        st_dataset_list = st_cfg.DATASETS.TRAIN
    else:
        tch_dataset_list = tch_cfg.DATASETS.TEST
        st_dataset_list = st_cfg.DATASETS.TEST
    datasets = build_dataset(tch_dataset_list, st_dataset_list, tch_cfg, tch_transform=tch_train_transform, st_transform=st_train_transform, target_transform=target_transform, is_train=is_train)

    shuffle = is_train or distributed

    data_loaders = []

    for dataset in datasets:
        if distributed:
            sampler = samplers.DistributedSampler(dataset, shuffle=shuffle)
        elif shuffle:
            sampler = torch.utils.data.RandomSampler(dataset)
        else:
            sampler = torch.utils.data.sampler.SequentialSampler(dataset)

        batch_size = tch_cfg.SOLVER.BATCH_SIZE if is_train else tch_cfg.TEST.BATCH_SIZE
        batch_sampler = torch.utils.data.sampler.BatchSampler(sampler=sampler, batch_size=batch_size, drop_last=is_train)
        if max_iter is not None:
            batch_sampler = samplers.IterationBasedBatchSampler(batch_sampler, num_iterations=max_iter, start_iter=start_iter)

        data_loader = DataLoader(dataset, num_workers=tch_cfg.DATA_LOADER.NUM_WORKERS, batch_sampler=batch_sampler,
                                 pin_memory=tch_cfg.DATA_LOADER.PIN_MEMORY, collate_fn=MultiScaleBatchCollator(is_train, batch_size) if tch_cfg.DATA_LOADER.MULTISCALE else BatchCollator(is_train,tch_cfg.MODEL.HEAD.NAME))
        data_loaders.append(data_loader)

    if is_train:
        # during training, a single (possibly concatenated) data_loader is returned
        assert len(data_loaders) == 1
        return data_loaders[0]
    return data_loaders

def make_test_data_loader(cfg, is_train=True, distributed=False, max_iter=None, start_iter=0):
    train_transform = build_transforms(cfg, is_train=is_train)
    target_transform = build_target_transform(cfg) if is_train else None
    dataset_list = cfg.DATASETS.TRAIN if is_train else cfg.DATASETS.TEST
    datasets = build_dataset(dataset_list, None, cfg, tch_transform=train_transform, st_transform=train_transform, target_transform=target_transform, is_train=is_train)

    shuffle = is_train or distributed

    data_loaders = []

    for dataset in datasets:
        if distributed:
            sampler = samplers.DistributedSampler(dataset, shuffle=shuffle)
        elif shuffle:
            sampler = torch.utils.data.RandomSampler(dataset)
        else:
            sampler = torch.utils.data.sampler.SequentialSampler(dataset)

        batch_size = cfg.SOLVER.BATCH_SIZE if is_train else cfg.TEST.BATCH_SIZE
        batch_sampler = torch.utils.data.sampler.BatchSampler(sampler=sampler, batch_size=batch_size, drop_last=is_train)
        if max_iter is not None:
            batch_sampler = samplers.IterationBasedBatchSampler(batch_sampler, num_iterations=max_iter, start_iter=start_iter)

        data_loader = DataLoader(dataset, num_workers=cfg.DATA_LOADER.NUM_WORKERS, batch_sampler=batch_sampler,
                                 pin_memory=cfg.DATA_LOADER.PIN_MEMORY, collate_fn=MultiScaleBatchCollator(is_train, batch_size) if cfg.DATA_LOADER.MULTISCALE else TestBatchCollator(is_train,cfg.MODEL.HEAD.NAME))
        data_loaders.append(data_loader)

    if is_train:
        # during training, a single (possibly concatenated) data_loader is returned
        assert len(data_loaders) == 1
        return data_loaders[0]
    return data_loaders
