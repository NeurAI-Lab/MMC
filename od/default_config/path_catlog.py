import os
from od.data.datasets.dataset_class_names import dataset_classes


class DatasetCatalog:
    DATA_DIR = '/input/datasets'
    DATASETS = {
        'voc_2007_train': {
            "data_dir": "VOC2007",
            "split": "train"
        },
        'voc_2007_val': {
            "data_dir": "VOC2007",
            "split": "val"
        },
        'voc_2007_trainval': {
            "data_dir": "VOC2007",
            "split": "trainval"
        },
        'voc_2007_test': {
            "data_dir": "VOC2007",
            "split": "test"
        },
        'voc_2007_test_temp': {
            "data_dir": "VOC2007",
            "split": "test_temp"
        },
        'voc_2012_train': {
            "data_dir": "VOC2012",
            "split": "train"
        },
        'voc_2012_val': {
            "data_dir": "VOC2012",
            "split": "val"
        },
        'voc_2012_trainval': {
            "data_dir": "VOC2012",
            "split": "trainval"
        },
        'voc_2012_test': {
            "data_dir": "VOC2012",
            "split": "test"
        },
        'coco_2014_valminusminival': {
            "data_dir": "val2014",
            "ann_file": "annotations/instances_valminusminival2014.json"
        },
        'coco_2014_minival': {
            "data_dir": "val2014",
            "ann_file": "annotations/instances_minival2014.json"
        },
        'coco_2014_train': {
            "data_dir": "train2014",
            "ann_file": "annotations/instances_train2014.json"
        },
        'coco_2014_val': {
            "data_dir": "val2014",
            "ann_file": "annotations/instances_val2014.json"
        },
        'bdd_2018_train': {
            "data_dir": "images/train2018",
            "ann_file": "annotations/instances_train2018.json"
        },
        'bdd_2018_val': {
            "data_dir": "images/val2018",
            "ann_file": "annotations/instances_val2018.json"
        },
        'bdd_day_coco_train': {
            "data_dir": "images/val2018",
            "ann_file": "annotations/day_instances_2class_val2018.json"
        },
        'bdd_night_coco_train': {
            "data_dir": "images/val2018",
            "ann_file": "annotations/night_instances_2class_val2018.json"
        },
        'cityscapes_val': {
            "data_dir": "",
            "ann_file": "gtFine/annotations_coco_format_v2/instances_val.json"
        },
        'had_2018_train': {
            "data_dir": "train",
            "ann_file": "annotations/had/instances_train2018.json"
        },
        'had_2018_val': {
            "data_dir": "test/positives",
            "ann_file": "annotations/had/instances_val2018.json"
        },
        'had_2018_minival': {
            "data_dir": "test/positives",
            "ann_file": "annotations/had/instances_val2018.json"
        },
        'ark_2020_train': {
            "data_dir": "train",
            "ann_file": "annotations/ark/instances_train2020.json"
        },
        'ark_2020_val': {
            "data_dir": "test/positives",
            "ann_file": "annotations/ark/instances_val2020.json"
        },
        'ark_2020_minival': {
            "data_dir": "test/positives",
            "ann_file": "annotations/ark/instances_minival2020.json"
        },
        'ark_2020_china_test': {
            "data_dir": "images",
            "ann_file": "annotations/instances_test_cn2020.json"
        },
        'flir_rgb_coco_train': {
            "data_dir": "train/RGB_new",
            "ann_file": "train/rgb_annotations_pseudo_bal.json"
        },
        'flir_rgb_coco_test': {
            "data_dir": "val/RGB_new",
            "ann_file": "val/rgb_annotations_pseudo_bal.json"
        },
        'flir_rgb_style_coco_test': {
            "data_dir": "val/RGB_new_stylized_1",
            "ann_file": "val/rgb_annotations_pseudo_bal.json"
        },
        'flir_rgb_fog_coco_test': {
            "data_dir": "val/RGB_new_fog",
            "ann_file": "val/rgb_annotations_pseudo_bal.json"
        },
        'flir_rgb_snow_coco_test': {
            "data_dir": "val/RGB_new_snow",
            "ann_file": "val/rgb_annotations_pseudo_bal.json"
        },
        'flir_rgb_gnoise_coco_test': {
            "data_dir": "val/RGB_new_gnoise",
            "ann_file": "val/rgb_annotations_pseudo_bal.json"
        },
        'flir_rgb_day_coco_test': {
            "data_dir": "val/RGB_new",
            "ann_file": "val/rgb_annotations_pseudo_bal_day.json"
        },
        'flir_rgb_night_coco_test': {
            "data_dir": "val/RGB_new",
            "ann_file": "val/rgb_annotations_pseudo_bal_night.json"
        },
        'flir_thermal_coco_train': {
            "data_dir": "train/",
            "ann_file": "train/thermal_annotations_new_bal.json"
        },
        'flir_thermal_coco_test': {
            "data_dir": "val/",
            "ann_file": "val/thermal_annotations_new_bal.json"
        },
        'flir_rgb_orig_coco_train': {
            "data_dir": "train/",
            "ann_file": "train/rgb_annotations_new.json"
        },
        'flir_rgb_orig_coco_test': {
            "data_dir": "val/",
            "ann_file": "val/rgb_annotations_new.json"
        },
        'flir_rgb_orig_coco_train_bal': {
            "data_dir": "train/",
            "ann_file": "train/rgb_annotations_orig_bal.json"
        },
        'flir_rgb_orig_coco_test_bal': {
            "data_dir": "val/",
            "ann_file": "val/rgb_annotations_orig_bal.json"
        },
        'flir_combined_coco_train_bal': {
            "data_dir": "train/",
            "ann_file": "train/combined_annotations_bal.json"
        },
        'flir_combined_coco_test_bal': {
            "data_dir": "val/",
            "ann_file": "val/combined_annotations_bal.json"
        },
        'kaist_rgb_coco_train_set05': {
            "data_dir": "train",
            "ann_file": "train/sanitized_annotations/annotations/instances_visible_train.json"
        },
        'kaist_thermal_coco_train_set05': {
            "data_dir": "train",
            "ann_file": "train/sanitized_annotations/annotations/instances_thm_train.json"
        },
        'kaist_combined_coco_train_set611': {
            "data_dir": "train",
            "ann_file": "train/sanitized_annotations/annotations/instances_comb.json"
        },
        'kaist_rgb_coco_test_set611': {
            "data_dir": "test",
            "ann_file": "test/sanitized_annotations/annotations/instances_visible_test.json"
        },
        'kaist_thermal_coco_test_set611': {
            "data_dir": "test",
            "ann_file": "test/sanitized_annotations/annotations/instances_thm_test.json"
        },
        'kaist_rgb_coco_test_day': {
            "data_dir": "test",
            "ann_file": "test/sanitized_annotations/annotations/instances_visible_day_test.json"
        },
        'kaist_rgb_coco_test_night': {
            "data_dir": "test",
            "ann_file": "test/sanitized_annotations/annotations/instances_visible_night_test.json"
        },
        'flir_rgb_coco_train_sane': {
            "data_dir": "train",
            "ann_file": "FLIR_aligned/aligned/align/train_rgb_sane1.json"
        },
        'flir_rgb_coco_test_sane': {
            "data_dir": "val",
            "ann_file": "FLIR_aligned/aligned/align/test_rgb_sane1.json"
        },
        'flir_rgb_corrupt_coco_test_sane': {
            "data_dir": "val",
            "ann_file": "FLIR_aligned/aligned/align/test_rgb_corrupt_sane.json"
        },
        'flir_thm_coco_train_sane': {
            "data_dir": "train",
            "ann_file": "FLIR_aligned/aligned/align/train_thm_sane1.json"
        },
        'flir_thm_coco_test_sane': {
            "data_dir": "val",
            "ann_file": "FLIR_aligned/aligned/align/test_thm_sane1.json"
        },
    }

    @staticmethod
    def get(name,dataset_path, data_dir="", ann_file=""):
        if "voc" in name:
            voc_root = DatasetCatalog.DATA_DIR
            if dataset_path!="":
                voc_root=dataset_path
            elif 'VOC_ROOT' in os.environ:
                voc_root = os.environ['VOC_ROOT']

            attrs = DatasetCatalog.DATASETS[name]
            args = dict(
                data_dir=os.path.join(voc_root, attrs["data_dir"]),
                split=attrs["split"],
            )
            return dict(factory="VOCDataset", args=args)
        elif (name in dataset for dataset in dataset_classes):
            coco_root = DatasetCatalog.DATA_DIR
            if dataset_path!="":
                coco_root=dataset_path
            elif 'COCO_ROOT' in os.environ:
                coco_root = os.environ['COCO_ROOT']

            attrs = DatasetCatalog.DATASETS[name]
            args = dict(
                data_dir=os.path.join(coco_root, data_dir) if data_dir else os.path.join(coco_root, attrs["data_dir"]),
                ann_file=os.path.join(coco_root, ann_file) if ann_file else os.path.join(coco_root, attrs["ann_file"]),
            )
            if 'test' in name or 'val' in name:
                return dict(factory="COCOTestDataset", args=args)
            else:
                return dict(factory="COCODataset", args=args)

        raise RuntimeError("Dataset not available: {}".format(name))
