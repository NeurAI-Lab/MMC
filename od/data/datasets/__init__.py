from torch.utils.data import ConcatDataset

from od.default_config.path_catlog import DatasetCatalog
from .voc import VOCDataset
from .coco import COCODataset, COCOTestDataset

_DATASETS = {
    'VOCDataset': VOCDataset,
    'COCODataset': COCODataset,
    'COCOTestDataset': COCOTestDataset,
}


def build_dataset(tch_dataset_list, st_dataset_list, cfg, tch_transform=None, st_transform=None, target_transform=None, is_train=True):
    assert len(tch_dataset_list) > 0
    datasets = []
    dataset_path=cfg.DATASETS.PATH
    data_dir = cfg.DATASETS.DATA_DIR
    ann_file = cfg.DATASETS.ANN_FILE
    for i in range(len(tch_dataset_list)):
        tch_dataset_name = tch_dataset_list[i]
        tch_data = DatasetCatalog.get(tch_dataset_name,dataset_path,data_dir, ann_file)
        factory = _DATASETS[tch_data['factory']]

        if factory == COCOTestDataset:
            args = tch_data['args']
        else:
            args = {}
            args['tch_data_dir'] = tch_data['args']['data_dir']
            args['tch_ann_file'] = tch_data['args']['ann_file']
            st_dataset_name = st_dataset_list[i]
            st_data = DatasetCatalog.get(st_dataset_name, dataset_path)
            args['st_data_dir'] = st_data['args']['data_dir']
            args['st_ann_file'] = st_data['args']['ann_file']

        args['include_background'] = cfg.DATA_LOADER.INCLUDE_BACKGROUND
        args['tch_transform'] = tch_transform
        args['st_transform'] = st_transform
        args['target_transform'] = target_transform
        if factory == VOCDataset:
            args['keep_difficult'] = not is_train
        elif factory == COCODataset:
            args['remove_empty'] = is_train
        dataset = factory(**args)
        datasets.append(dataset)
    # for testing, return a list of datasets
    if not is_train:
        return datasets
    dataset = datasets[0]
    if len(datasets) > 1:
        dataset = ConcatDataset(datasets)

    return [dataset]