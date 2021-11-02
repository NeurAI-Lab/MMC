import os
import torch.utils.data
import numpy as np
from PIL import Image

from od.structures.container import Container


class COCODataset(torch.utils.data.Dataset):
    def __init__(self, tch_data_dir, tch_ann_file, st_data_dir, st_ann_file, include_background=True, tch_transform=None, st_transform=None, target_transform=None, remove_empty=False):
        from pycocotools.coco import COCO
        self.cocotch = COCO(tch_ann_file)
        self.tch_data_dir = tch_data_dir
        self.cocost = COCO(st_ann_file)
        self.st_data_dir = st_data_dir
        self.tch_transform = tch_transform
        self.st_transform = st_transform
        self.target_transform = target_transform
        self.remove_empty = remove_empty
        if self.remove_empty:
            # when training, images without annotations are removed.
            self.tch_ids = list(self.cocotch.imgToAnns.keys())
            self.st_ids = list(self.cocost.imgToAnns.keys())
        else:
            # when testing, all images used.
            self.tch_ids = list(self.cocotch.imgs.keys())
            self.st_ids = list(self.cocost.imgs.keys())

        self.include_background = include_background
        coco_categories = sorted(self.cocotch.getCatIds())

        # CenterNet and YOLO v2/v3 will use the IF statement. SSD uses the ELSE statement.
        if not self.include_background:
            self.coco_id_to_contiguous_id = {coco_id: i for i, coco_id in enumerate(coco_categories)}
        else:
            self.coco_id_to_contiguous_id = {coco_id: i+1 for i, coco_id in enumerate(coco_categories)}
        self.contiguous_id_to_coco_id = {v: k for k, v in self.coco_id_to_contiguous_id.items()}

    def __getitem__(self, index):
        tch_image_id = self.tch_ids[index]
        tch_boxes, tch_labels = self._get_annotation(tch_image_id, teacher=True)
        tch_image = self._read_image(tch_image_id, teacher=True)
        st_image_id = self.st_ids[index]
        st_boxes, st_labels = self._get_annotation(st_image_id, teacher=False)
        st_image = self._read_image(st_image_id, teacher=False)

        #display ground truth
        # for ii in range(len(tch_boxes)):
        #     x1, y1, x2, y2 = tch_boxes[ii]
        #     cat = int(tch_labels[ii])
        #     color = ((10 * cat) % 256, (20 * cat) % 256, (5 * cat) % 256)
        #     st = (int(x1), int(y1))
        #     end = (int(x2), int(y2))
        #     cv2.rectangle(tch_image, st, end, color, 5)
        # dir = "dl"
        # out = rf"{dir}/{tch_image_id}.jpg"
        # cv2.imwrite(out, tch_image)
        # for ii in range(len(st_boxes)):
        #     x1, y1, x2, y2 = st_boxes[ii]
        #     cat = int(st_labels[ii])
        #     color = ((10 * cat) % 256, (20 * cat) % 256, (5 * cat) % 256)
        #     st = (int(x1), int(y1))
        #     end = (int(x2), int(y2))
        #     cv2.rectangle(st_image, st, end, color, 5)
        # dir = "dl"
        # out = rf"{dir}/{st_image_id}.jpg"
        # cv2.imwrite(out, st_image)

        # teacher
        if self.tch_transform:
            tch_image, tch_boxes, tch_labels = self.tch_transform(tch_image, tch_boxes, tch_labels)
        if self.target_transform:
            tch_boxes, tch_labels = self.target_transform(tch_boxes, tch_labels)
        tch_targets = Container(
            boxes=tch_boxes,
            labels=tch_labels,
        )
        #student
        if self.st_transform:
            st_image, st_boxes, st_labels = self.st_transform(st_image, st_boxes, st_labels)
        if self.target_transform:
            st_boxes, st_labels = self.target_transform(st_boxes, st_labels)
        st_targets = Container(
            boxes=st_boxes,
            labels=st_labels,
        )
        return (tch_image, tch_targets, index), (st_image, st_targets, index)

    def get_annotation(self, index):
        image_id = self.ids[index]
        return image_id, self._get_annotation(image_id)

    def __len__(self):
        return len(self.tch_ids)

    def _get_annotation(self, image_id, teacher=True):

        if teacher:
            ann_ids = self.cocotch.getAnnIds(imgIds=image_id)
            ann = self.cocotch.loadAnns(ann_ids)
        else:
            ann_ids = self.cocost.getAnnIds(imgIds=image_id)
            ann = self.cocost.loadAnns(ann_ids)
        # filter crowd annotations
        ann = [obj for obj in ann if obj["iscrowd"] == 0]
        boxes = np.array([self._xywh2xyxy(obj["bbox"]) for obj in ann], np.float32).reshape((-1, 4))
        labels = np.array([self.coco_id_to_contiguous_id[obj["category_id"]] for obj in ann], np.int64).reshape((-1,))
        # remove invalid boxes
        keep = (boxes[:, 3] > boxes[:, 1]) & (boxes[:, 2] > boxes[:, 0])
        boxes = boxes[keep]
        labels = labels[keep]
        return boxes, labels

    def _xywh2xyxy(self, box):
        x1, y1, w, h = box
        return [x1, y1, x1 + w, y1 + h]

    def get_img_info(self, index):
        image_id = self.ids[index]
        img_data = self.coco.imgs[image_id]
        return img_data

    def _read_image(self, image_id, teacher=True):
        if teacher:
            file_name = self.cocotch.loadImgs(image_id)[0]['file_name']
            if file_name.startswith('/'):
                file_name = file_name[1:]
            image_file = os.path.join(self.tch_data_dir, file_name)
        else:
            file_name = self.cocost.loadImgs(image_id)[0]['file_name']
            if file_name.startswith('/'):
                file_name = file_name[1:]
            image_file = os.path.join(self.st_data_dir, file_name)

        image = Image.open(image_file).convert("RGB")
        image = np.array(image)
        return image

    def get_image_path(self, img_id):
        coco = self.coco
        img_metadata = coco.loadImgs(img_id)[0]
        path = img_metadata['file_name']
        return os.path.join(self.data_dir, path)

class COCOTestDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, ann_file, include_background=True, tch_transform=None, st_transform=None, target_transform=None, remove_empty=False):
        from pycocotools.coco import COCO
        self.coco = COCO(ann_file)
        self.data_dir = data_dir
        self.transform = tch_transform
        self.target_transform = target_transform
        self.remove_empty = remove_empty
        if self.remove_empty:
            # when training, images without annotations are removed.
            self.ids = list(self.coco.imgToAnns.keys())
        else:
            # when testing, all images used.
            self.ids = list(self.coco.imgs.keys())
        self.include_background = include_background
        coco_categories = sorted(self.coco.getCatIds())

        # CenterNet and YOLO v2/v3 will use the IF statement. SSD uses the ELSE statement.
        if not self.include_background:
            self.coco_id_to_contiguous_id = {coco_id: i for i, coco_id in enumerate(coco_categories)}
        else:
            self.coco_id_to_contiguous_id = {coco_id: i+1 for i, coco_id in enumerate(coco_categories)}
        self.contiguous_id_to_coco_id = {v: k for k, v in self.coco_id_to_contiguous_id.items()}

    def __getitem__(self, index):
        image_id = self.ids[index]
        boxes, labels = self._get_annotation(image_id)
        image = self._read_image(image_id)

        # for ii in range(len(boxes)):
        #     x1, y1, x2, y2 = boxes[ii]
        #     cat = int(labels[ii])
        #     color = ((10 * cat) % 256, (20 * cat) % 256, (5 * cat) % 256)
        #     st = (int(x1), int(y1))
        #     end = (int(x2), int(y2))
        #     cv2.rectangle(image, st, end, color, 5)
        # dir = "train"
        # out = rf"{dir}/{image_id}.jpg"
        # cv2.imwrite(out, image)

        if self.transform:
            image, boxes, labels = self.transform(image, boxes, labels)
        if self.target_transform:
            boxes, labels = self.target_transform(boxes, labels)
        targets = Container(
            boxes=boxes,
            labels=labels,
        )
        return image, targets, index

    def get_annotation(self, index):
        image_id = self.ids[index]
        return image_id, self._get_annotation(image_id)

    def __len__(self):
        return len(self.ids)

    def _get_annotation(self, image_id):
        ann_ids = self.coco.getAnnIds(imgIds=image_id)
        ann = self.coco.loadAnns(ann_ids)
        # filter crowd annotations
        ann = [obj for obj in ann if obj["iscrowd"] == 0]
        boxes = np.array([self._xywh2xyxy(obj["bbox"]) for obj in ann], np.float32).reshape((-1, 4))
        labels = np.array([self.coco_id_to_contiguous_id[obj["category_id"]] for obj in ann], np.int64).reshape((-1,))
        # remove invalid boxes
        keep = (boxes[:, 3] > boxes[:, 1]) & (boxes[:, 2] > boxes[:, 0])
        boxes = boxes[keep]
        labels = labels[keep]
        return boxes, labels

    def _xywh2xyxy(self, box):
        x1, y1, w, h = box
        return [x1, y1, x1 + w, y1 + h]

    def get_img_info(self, index):
        image_id = self.ids[index]
        img_data = self.coco.imgs[image_id]
        return img_data

    def _read_image(self, image_id):
        file_name = self.coco.loadImgs(image_id)[0]['file_name']
        if file_name.startswith('/'):
            file_name = file_name[1:]
        image_file = os.path.join(self.data_dir, file_name)
        image = Image.open(image_file).convert("RGB")
        image = np.array(image)
        return image

    def get_image_path(self, img_id):
        coco = self.coco
        img_metadata = coco.loadImgs(img_id)[0]
        path = img_metadata['file_name']
        return os.path.join(self.data_dir, path)
