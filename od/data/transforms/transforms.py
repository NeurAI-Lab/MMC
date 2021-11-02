# from https://github.com/amdegroot/ssd.pytorch


import torch
import cv2
import numpy as np
import types
from PIL import Image, ImageDraw
from torchvision import transforms
from numpy import random
import math

def intersect(box_a, box_b):
    max_xy = np.minimum(box_a[:, 2:], box_b[2:])
    min_xy = np.maximum(box_a[:, :2], box_b[:2])
    inter = np.clip((max_xy - min_xy), a_min=0, a_max=np.inf)
    return inter[:, 0] * inter[:, 1]


def jaccard_numpy(box_a, box_b):
    """Compute the jaccard overlap of two sets of boxes.  The jaccard overlap
    is simply the intersection over union of two boxes.
    E.g.:
        A ∩ B / A ∪ B = A ∩ B / (area(A) + area(B) - A ∩ B)
    Args:
        box_a: Multiple bounding boxes, Shape: [num_boxes,4]
        box_b: Single bounding box, Shape: [4]
    Return:
        jaccard overlap: Shape: [box_a.shape[0], box_a.shape[1]]
    """
    inter = intersect(box_a, box_b)
    area_a = ((box_a[:, 2] - box_a[:, 0]) *
              (box_a[:, 3] - box_a[:, 1]))  # [A,B]
    area_b = ((box_b[2] - box_b[0]) *
              (box_b[3] - box_b[1]))  # [A,B]
    union = area_a + area_b - inter
    return inter / union  # [A,B]


class Compose(object):
    """Composes several augmentations together.
    Args:
        transforms (List[Transform]): list of transforms to compose.
    Example:
        >>> augmentations.Compose([
        >>>     transforms.CenterCrop(10),
        >>>     transforms.ToTensor(),
        >>> ])
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, boxes=None, labels=None):
        for t in self.transforms:
            img, boxes, labels = t(img, boxes, labels)
        return img, boxes, labels


class Lambda(object):
    """Applies a lambda as a transform."""

    def __init__(self, lambd):
        assert isinstance(lambd, types.LambdaType)
        self.lambd = lambd

    def __call__(self, img, boxes=None, labels=None):
        return self.lambd(img, boxes, labels)


class ToGrayscale(object):
    def __call__(self, image, boxes=None, labels=None):
        def rgb2gray(rgb):
            r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
            gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
            return gray

        image_grayscale = rgb2gray(image)
        image_grayscale = np.repeat(image_grayscale[..., np.newaxis], 3, axis=-1)
        return image_grayscale, boxes, labels

class JPEG_Encode(object):
    def __call__(self, image, boxes=None, labels=None):
        quality = random.randint(10,80)
        image_encoded = io.encode_jpeg(image, quality=quality)
        return image_encoded, boxes, labels

class JPEG_Decode(object):
    def __call__(self, image, boxes=None, labels=None):
        image_decoded = io.decode_image(input = image)
        return image_decoded, boxes, labels

class GaussianBlur(object):
    def __init__(self, kernel_size=(3,3), std_dev=1.5):
        self.kernel_size = kernel_size
        self.std_dev = std_dev

    def __call__(self, image, boxes=None, labels=None):
        image_blur = cv2.GaussianBlur(image, self.kernel_size, self.std_dev)
        return image_blur, boxes, labels

class ConvertFromInts(object):
    def __call__(self, image, boxes=None, labels=None):
        return image.astype(np.float32), boxes, labels


class SubtractMeans(object):
    def __init__(self, mean):
        self.mean = np.array(mean, dtype=np.float32)

    def __call__(self, image, boxes=None, labels=None):
        image = image.astype(np.float32)
        image -= self.mean
        return image.astype(np.float32), boxes, labels


class Standardize(object):
    def __init__(self, mean=None, std=None):
        self.mean = mean
        if self.mean is None:
            self.mean = np.array([[[0.485, 0.456, 0.406]]])
        self.std = std
        if self.std is None:
            self.std = np.array([[[0.229, 0.224, 0.225]]])

    def __call__(self, image, boxes=None, labels=None):
        image = (image.astype(np.float32)/255 - self.mean) / self.std
        return image.astype(np.float32), boxes, labels


class ToAbsoluteCoords(object):
    def __call__(self, image, boxes=None, labels=None):
        height, width, channels = image.shape
        boxes[:, 0] *= width
        boxes[:, 2] *= width
        boxes[:, 1] *= height
        boxes[:, 3] *= height

        return image, boxes, labels


class ToPercentCoords(object):
    def __call__(self, image, boxes=None, labels=None):
        height, width, channels = image.shape
        if (boxes.size):
            boxes[:, 0] /= width
            boxes[:, 2] /= width
            boxes[:, 1] /= height
            boxes[:, 3] /= height

        return image, boxes, labels


class Resize(object):
    def __init__(self, size=300):
        self.size = size

    def __call__(self, image, boxes=None, labels=None):
        image = cv2.resize(image, (self.size,
                                   self.size))
        return image, boxes, labels

class UpDownSample(object):
    def __init__(self, size1=1024, size2=512):
        self.size1 = size1
        self.size2 = size2

    def __call__(self, image, boxes=None, labels=None):
        image = cv2.resize(image, (self.size1,
                                   self.size1), cv2.INTER_NEAREST)
        image = cv2.resize(image, (self.size2,
                                   self.size2), cv2.INTER_NEAREST)
        return image, boxes, labels

class Remake(object):
    def __init__(self, size=300):
        self.size = size

    def __call__(self, image, boxes=None, labels=None):
        a=[]
        l=[]
        if boxes is not None:
            if len(boxes)!=0:
                for i,s in enumerate(boxes):
                        boxes[i][:]= boxes[i][:]*self.size
                        if boxes[i][0]<self.size and boxes[i][1]<self.size :
                            boxes[i][2],boxes[i][3]=min(boxes[i][2],self.size ), min(boxes[i][3],self.size )
                            a.append(boxes[i])
                            l.append(labels[i])
                boxes=a
                labels=l
        # test_plot_boxes(image, boxes, "/volumes2/tasks/ARL-1186-Transformers", "after.jpg")
        return image, boxes, labels

class RandomSaturation(object):
    def __init__(self, lower=0.5, upper=1.5):
        self.lower = lower
        self.upper = upper
        assert self.upper >= self.lower, "contrast upper must be >= lower."
        assert self.lower >= 0, "contrast lower must be non-negative."

    def __call__(self, image, boxes=None, labels=None):
        if random.randint(2):
            image[:, :, 1] *= random.uniform(self.lower, self.upper)

        return image, boxes, labels


class RandomHue(object):
    def __init__(self, delta=18.0):
        assert delta >= 0.0 and delta <= 360.0
        self.delta = delta

    def __call__(self, image, boxes=None, labels=None):
        if random.randint(2):
            image[:, :, 0] += random.uniform(-self.delta, self.delta)
            image[:, :, 0][image[:, :, 0] > 360.0] -= 360.0
            image[:, :, 0][image[:, :, 0] < 0.0] += 360.0
        return image, boxes, labels


class RandomLightingNoise(object):
    def __init__(self):
        self.perms = ((0, 1, 2), (0, 2, 1),
                      (1, 0, 2), (1, 2, 0),
                      (2, 0, 1), (2, 1, 0))

    def __call__(self, image, boxes=None, labels=None):
        if random.randint(2):
            swap = self.perms[random.randint(len(self.perms))]
            shuffle = SwapChannels(swap)  # shuffle channels
            image = shuffle(image)
        return image, boxes, labels


class ConvertColor(object):
    def __init__(self, current, transform):
        self.transform = transform
        self.current = current

    def __call__(self, image, boxes=None, labels=None):
        if self.current == 'BGR' and self.transform == 'HSV':
            image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        elif self.current == 'RGB' and self.transform == 'HSV':
            image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        elif self.current == 'BGR' and self.transform == 'RGB':
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        elif self.current == 'HSV' and self.transform == 'BGR':
            image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)
        elif self.current == 'HSV' and self.transform == "RGB":
            image = cv2.cvtColor(image, cv2.COLOR_HSV2RGB)
        else:
            raise NotImplementedError
        return image, boxes, labels


class RandomContrast(object):
    def __init__(self, lower=0.5, upper=1.5):
        self.lower = lower
        self.upper = upper
        assert self.upper >= self.lower, "contrast upper must be >= lower."
        assert self.lower >= 0, "contrast lower must be non-negative."

    # expects float image
    def __call__(self, image, boxes=None, labels=None):
        if random.randint(2):
            alpha = random.uniform(self.lower, self.upper)
            image *= alpha
        return image, boxes, labels


class RandomBrightness(object):
    def __init__(self, delta=32):
        assert delta >= 0.0
        assert delta <= 255.0
        self.delta = delta

    def __call__(self, image, boxes=None, labels=None):
        if random.randint(2):
            delta = random.uniform(-self.delta, self.delta)
            image += delta
        return image, boxes, labels


class ToCV2Image(object):
    def __call__(self, tensor, boxes=None, labels=None):
        return tensor.cpu().numpy().astype(np.float32).transpose((1, 2, 0)), boxes, labels


class ToTensor(object):
    def __call__(self, cvimage, boxes=None, labels=None):
        return torch.from_numpy(cvimage.astype(np.float32)).permute(2, 0, 1), boxes, labels


class RandomSampleCrop(object):
    """Crop
    Arguments:
        img (Image): the image being input during training
        boxes (Tensor): the original bounding boxes in pt form
        labels (Tensor): the class labels for each bbox
        mode (float tuple): the min and max jaccard overlaps
    Return:
        (img, boxes, classes)
            img (Image): the cropped image
            boxes (Tensor): the adjusted bounding boxes in pt form
            labels (Tensor): the class labels for each bbox
    """

    def __init__(self):
        self.sample_options = (
            # using entire original input image
            None,
            # sample a patch s.t. MIN jaccard w/ obj in .1,.3,.4,.7,.9
            (0.1, None),
            (0.3, None),
            (0.7, None),
            (0.9, None),
            # randomly sample a patch
            (None, None),
        )

    def __call__(self, image, boxes=None, labels=None):
        # guard against no boxes
        if boxes is not None and boxes.shape[0] == 0:
            return image, boxes, labels
        height, width, _ = image.shape
        while True:
            # randomly choose a mode
            mode = random.choice(self.sample_options)
            if mode is None:
                return image, boxes, labels

            min_iou, max_iou = mode
            if min_iou is None:
                min_iou = float('-inf')
            if max_iou is None:
                max_iou = float('inf')

            # max trails (50)
            for _ in range(50):
                current_image = image

                w = random.uniform(0.3 * width, width)
                h = random.uniform(0.3 * height, height)

                # aspect ratio constraint b/t .5 & 2
                if h / w < 0.5 or h / w > 2:
                    continue

                left = random.uniform(width - w)
                top = random.uniform(height - h)

                # convert to integer rect x1,y1,x2,y2
                rect = np.array([int(left), int(top), int(left + w), int(top + h)])

                # calculate IoU (jaccard overlap) b/t the cropped and gt boxes
                overlap = jaccard_numpy(boxes, rect)

                # is min and max overlap constraint satisfied? if not try again
                if overlap.max() < min_iou or overlap.min() > max_iou:
                    continue

                # cut the crop from the image
                current_image = current_image[rect[1]:rect[3], rect[0]:rect[2],
                                :]

                # keep overlap with gt box IF center in sampled patch
                centers = (boxes[:, :2] + boxes[:, 2:]) / 2.0

                # mask in all gt boxes that above and to the left of centers
                m1 = (rect[0] < centers[:, 0]) * (rect[1] < centers[:, 1])

                # mask in all gt boxes that under and to the right of centers
                m2 = (rect[2] > centers[:, 0]) * (rect[3] > centers[:, 1])

                # mask in that both m1 and m2 are true
                mask = m1 * m2

                # have any valid boxes? try again if not
                if not mask.any():
                    continue

                # take only matching gt boxes
                current_boxes = boxes[mask, :].copy()

                # take only matching gt labels
                current_labels = labels[mask]

                # should we use the box left and top corner or the crop's
                current_boxes[:, :2] = np.maximum(current_boxes[:, :2],
                                                  rect[:2])
                # adjust to crop (by substracting crop's left,top)
                current_boxes[:, :2] -= rect[:2]

                current_boxes[:, 2:] = np.minimum(current_boxes[:, 2:],
                                                  rect[2:])
                # adjust to crop (by substracting crop's left,top)
                current_boxes[:, 2:] -= rect[:2]

                return current_image, current_boxes, current_labels


class Expand(object):
    def __init__(self, mean):
        self.mean = mean

    def __call__(self, image, boxes, labels):
        if random.randint(2):
            return image, boxes, labels

        height, width, depth = image.shape
        ratio = random.uniform(1, 4)
        left = random.uniform(0, width * ratio - width)
        top = random.uniform(0, height * ratio - height)

        expand_image = np.zeros(
            (int(height * ratio), int(width * ratio), depth),
            dtype=image.dtype)
        expand_image[:, :, :] = self.mean
        expand_image[int(top):int(top + height),
        int(left):int(left + width)] = image
        image = expand_image
        if boxes is not None:
            if len(boxes)!=0:
                boxes = boxes.copy()
                boxes[:, :2] += (int(left), int(top))
                boxes[:, 2:] += (int(left), int(top))

        return image, boxes, labels


class RandomMirror(object):
    def __call__(self, image, boxes, classes):
        _, width, _ = image.shape
        if boxes is not None:
            if len(boxes)!=0:
                if random.randint(2):
                    image = image[:, ::-1]
                    boxes = boxes.copy()
                    boxes[:, 0::2] = width - boxes[:, 2::-2]
        return image, boxes, classes


class Flip(object):
    def __call__(self, image, boxes, classes):
        '''
            Flip image horizontally.
            image: a PIL image
            boxes: Bounding boxes, a tensor of dimensions (#objects, 4)
        '''
        new_image = F.hflip(image)

        # flip boxes
        new_boxes = boxes.clone()
        new_boxes[:, 0] = image.width - boxes[:, 0]
        new_boxes[:, 2] = image.width - boxes[:, 2]
        new_boxes = new_boxes[:, [2, 1, 0, 3]]
        return new_image, new_boxes, classes


class SwapChannels(object):
    """Transforms a tensorized image by swapping the channels in the order
     specified in the swap tuple.
    Args:
        swaps (int triple): final order of channels
            eg: (2, 1, 0)
    """

    def __init__(self, swaps):
        self.swaps = swaps

    def __call__(self, image):
        """
        Args:
            image (Tensor): image tensor to be transformed
        Return:
            a tensor with channels swapped according to swap
        """
        # if torch.is_tensor(image):
        #     image = image.data.cpu().numpy()
        # else:
        #     image = np.array(image)
        image = image[:, :, self.swaps]
        return image


class PhotometricDistort(object):
    def __init__(self):
        self.pd = [
            RandomContrast(),  # RGB
            ConvertColor(current="RGB", transform='HSV'),  # HSV
            RandomSaturation(),  # HSV
            RandomHue(),  # HSV
            ConvertColor(current='HSV', transform='RGB'),  # RGB
            RandomContrast()  # RGB
        ]
        self.rand_brightness = RandomBrightness()
        self.rand_light_noise = RandomLightingNoise()

    def __call__(self, image, boxes, labels):
        im = image.copy()
        im, boxes, labels = self.rand_brightness(im, boxes, labels)
        if random.randint(2):
            distort = Compose(self.pd[:-1])
        else:
            distort = Compose(self.pd[1:])
        im, boxes, labels = distort(im, boxes, labels)
        return self.rand_light_noise(im, boxes, labels)


class ImageDescription(object):
    def __init__(self, cfg):
        self.batch_size = cfg.SOLVER.BATCH_SIZE
        self.multi_scale_step = cfg.INPUT.MULTI_SCALE_STEP
        self.input_sizes = np.array(cfg.INPUT.SCALES)
        self.current_size = None
        self.counter = 0
        self.scale_x = None
        self.scale_y = None
        self.it = iter(self.input_sizes)

    def __call__(self, image, boxes=None, labels=None):
        self.scale_x = float(self.current_size) / float(image.shape[1])
        self.scale_y = float(self.current_size) / float(image.shape[0])

    def next(self):
        if self.counter == 0:
            self.counter = self.batch_size * self.multi_scale_step - 1
            try:
                self.current_size = next(self.it)
            except:
                self.it = iter(self.input_sizes)
                self.current_size = next(self.it)
        else:
            self.counter -= 1

class ResizeImageBoxes(object):
    def __init__(self, cfg, is_train):
        self.is_train=is_train
        self.test_size = cfg.INPUT.IMAGE_SIZE
        self.image_description = ImageDescription(cfg)

    def __call__(self, image, boxes=None, labels=None):
        if self.is_train:
            assert boxes is not None
            self.image_description.next()
            self.image_description(image,boxes,labels)
            scale_height = self.image_description.scale_y
            scale_width = self.image_description.scale_x

            boxes[:, 0::2] *= scale_width
            boxes[:, 1::2] *= scale_height
        else:
            scale_height = self.test_size / image.shape[0]
            scale_width = self.test_size / image.shape[1]

        image = cv2.resize(image, None, None, fx=scale_width, fy=scale_height, interpolation=cv2.INTER_LINEAR)

        return image, boxes, labels

class HSVDistortYolo(object):
    def __init__(self):
        self.hue = 0.1
        self.saturation = 1.5
        self.exposure = 1.5

    def __call__(self, image, boxes, labels):

        def rand_scale(s):
            scale = random.uniform(1, s)
            if (random.randint(1, 10000) % 2):
                return scale
            return 1. / scale

        hue = random.uniform(-self.hue, self.hue)
        sat = rand_scale(self.saturation)
        val = rand_scale(self.exposure)

        # ---- in PIL format (Yolo Repo)--------------
        im = image.convert('HSV')
        cs = list(im.split())
        cs[1] = cs[1].point(lambda i: min(255, max(0, int(i*sat))))
        cs[2] = cs[2].point(lambda i: min(255, max(0, int(i*val))))
        def change_hue(x):
            x += hue * 255
            if x > 255:
                x -= 255
            if x < 0:
                x += 255
            return x
        cs[0] = cs[0].point(change_hue)
        im = Image.merge(im.mode, tuple(cs))
        image = im.convert('RGB')
        # -----------------------------------------

        return image, boxes, labels

class RandomFlip(object):
    """ Randomly flip image """
    def __init__(self):
        self.threshold = 0.5
        self.flip = False

    def __call__(self, image, boxes, labels):

        self.flip = random.random() < self.threshold
        im_w = image.shape[1]

        if self.flip:
            current_image = Image.fromarray(image, 'RGB')
            flipped_image = current_image.transpose(Image.FLIP_LEFT_RIGHT)
            current_boxes = []
            for box in boxes:
                x1, y1, x2, y2 = box
                w = x2 - x1
                anno_width = x2 - x1
                x1 = im_w - x1 - anno_width
                x2 = x1+w
                current_boxes.append([x1,y1,x2,y2])

            current_boxes = np.array(current_boxes, dtype=np.float32)
            flipped_image = np.array(flipped_image)
            return flipped_image, current_boxes, labels

        return image, boxes, labels

class BbAug(object):
    def __init__(self, policy_version):
        from bbaug.policies import policies
        # policy selection guide:
        # https://github.com/tensorflow/tpu/blob/master/models/official/detection/utils/autoaugment_utils.py
        if policy_version == 0:
            aug_policy = policies.policies_v0()
        elif policy_version == 1:
            aug_policy = policies.policies_v1()
        elif policy_version == 2:
            aug_policy = policies.policies_v2()
        elif policy_version == 3:
            aug_policy = policies.policies_v3()
        else:
            raise ValueError(f"Policy {policy_version} is not available. "
                             f"Only 4 versions of augmentation policy is available: v0, v1, v2 and v3")
        self.policy_container = policies.PolicyContainer(aug_policy)

    def __call__(self, image, boxes, labels=None):
        random_policy = self.policy_container.select_random_policy()
        # print(random_policy)
        # test_plot_boxes(image, boxes, "/volumes2/tasks/ARL-1186-Transformers", "before.jpg")
        # image = image.astype(np.uint8)
        a_image, a_boxes = self.policy_container.apply_augmentation(random_policy, image, boxes, labels)
        if len(a_boxes) > 0:
            a_labels = a_boxes[:, 0]
            a_boxes = a_boxes[:, 1:]
            a_boxes = a_boxes.astype(np.float32)
        else:
            a_labels = a_boxes
        # test_plot_boxes(image, boxes, "/volumes2/tasks/ARL-1186-Transformers", "after.jpg")
        # image = image.astype(np.float32)
        return a_image, a_boxes, a_labels


class Translate_X(object):
    def __init__(self, magnitude=50):
        import imgaug.augmenters as iaa
        self.aug = iaa.Affine(translate_px={'x': magnitude})

    def __call__(self, image, boxes, labels=None):
        test_plot_boxes(image, boxes, "/volumes2/tasks/ARL-1186-Transformers", "before.jpg")
        boxes = np.expand_dims(boxes, 0)
        image, boxes = self.aug(image=image, bounding_boxes=boxes)
        boxes = np.squeeze(boxes, 0)
        test_plot_boxes(image, boxes, "/volumes2/tasks/ARL-1186-Transformers", "after.jpg")
        return image, boxes, labels

#---------------------------------
# The augmentation from Yolo Repo
# Done usign PIL, not numpy
# Keeping for backup
#---------------------------------
class YoloCropOld(object):
    def __init__(self, size=416):
        self.jitter = 0.3
        self.shape = (size,size)
        self.hsv_distort_yolo = HSVDistortYolo()

    def __call__(self, image, boxes=None, labels=None):
        # guard against no boxes
        current_image = Image.fromarray(image,'RGB')
        oh = current_image.height
        ow = current_image.width

        dw = int(ow * self.jitter)
        dh = int(oh * self.jitter)

        pleft = random.randint(-dw, dw)
        pright = random.randint(-dw, dw)
        ptop = random.randint(-dh, dh)
        pbot = random.randint(-dh, dh)

        swidth = ow - pleft - pright
        sheight = oh - ptop - pbot

        sx = float(swidth) / ow
        sy = float(sheight) / oh

        flip = np.random.randint(2)

        cropped = current_image.crop((pleft, ptop, pleft + swidth - 1, ptop + sheight - 1))

        dx = (float(pleft) / ow) / sx
        dy = (float(ptop) / oh) / sy

        sized = cropped.resize(self.shape)

        if flip:
            sized = sized.transpose(Image.FLIP_LEFT_RIGHT)
        sized, _, _ = self.hsv_distort_yolo(sized, boxes, labels)

        current_boxes = []
        for box in boxes:
            x1, y1, x2, y2 = box
            x1 = min(0.999, max(0, x1 / sx - dx))
            y1 = min(0.999, max(0, y1 / sy - dy))
            x2 = min(0.999, max(0, x2 / sx - dx))
            y2 = min(0.999, max(0, y2 / sy - dy))

            c1 = (x1 + x2) / 2
            c2 = (y1 + y2) / 2
            w = (x2 - x1)
            h = (y2 - y1)

            if flip:
                c1 = 0.999 - c1

            if w < 0.001 or h < 0.001:
                continue

            x1 = c1 - w/2
            y1 = c2 - h/2
            x2 = c1 + w/2
            y2 = c2 + h/2
            current_boxes.append([x1,y1,x2,y2])

        current_image = np.array(sized)
        current_boxes = np.array(current_boxes,dtype=np.float32)

        return current_image.astype(np.float32), current_boxes, labels

#---------------------------------
# The augmentation from Yolo Tencent Repo
#---------------------------------
class YoloCrop(object):
    def __init__(self, size=416):
        self.jitter = 0.3
        self.shape = (size,size)
        self.output_w = size
        self.output_h = size
        self.hsv_distort_yolo = HSVDistortYolo()
        self.fill_color = 127

    def __call__(self, image, boxes=None, labels=None):

        import random

        # guard against no boxes
        current_image = Image.fromarray(image,'RGB')

        oh = current_image.height
        ow = current_image.width
        img_np = np.array(current_image)
        channels = img_np.shape[2] if len(img_np.shape) > 2 else 1
        dw = int(ow * self.jitter)
        dh = int(oh * self.jitter)

        new_ar = float(ow + random.randint(-dw, dw)) / (oh + random.randint(-dh, dh))
        scale = random.random()*(2-0.25) + 0.25
        if new_ar < 1:
            nh = int(scale * oh)
            nw = int(nh * new_ar)
        else:
            nw = int(scale * ow)
            nh = int(nw / new_ar)

        if self.output_w > nw:
            dx = random.randint(0, self.output_w - nw)
        else:
            dx = random.randint(self.output_w - nw, 0)

        if self.output_h > nh:
            dy = random.randint(0, self.output_h - nh)
        else:
            dy = random.randint(self.output_h - nh, 0)
        nxmin = max(0, -dx)
        nymin = max(0, -dy)
        nxmax = min(nw, -dx + self.output_w - 1)
        nymax = min(nh, -dy + self.output_h - 1)
        sx, sy = float(ow)/nw, float(oh)/nh
        orig_xmin = int(nxmin * sx)
        orig_ymin = int(nymin * sy)
        orig_xmax = int(nxmax * sx)
        orig_ymax = int(nymax * sy)
        orig_crop = current_image.crop((orig_xmin, orig_ymin, orig_xmax, orig_ymax))
        orig_crop_resize = orig_crop.resize((nxmax - nxmin, nymax - nymin))
        output_img = Image.new(current_image.mode, (self.output_w, self.output_h), color=(self.fill_color,)*channels)
        output_img.paste(orig_crop_resize, (0, 0))

        # flip = np.random.randint(2)
        # if flip:
        #     output_img = output_img.transpose(Image.FLIP_LEFT_RIGHT)
        output_img, _, _ = self.hsv_distort_yolo(output_img, boxes, labels)

        current_boxes = []
        for box in boxes:
            x1, y1, x2, y2 = box

            x1n = max(nxmin, int(x1/sx))
            x2n = min(nxmax, int(x2/sx))
            y1n = max(nymin, int(y1/sy))
            y2n = min(nymax, int(y2/sy))
            w = x2n-x1n
            h = y2n-y1n

            if w <= 2 or h <= 2:
                continue

            x1n = x1n - nxmin
            y1n = y1n - nymin
            # anno_width = x2-x1
            # if flip:
            #     x1n = current_image.size[0] - x1n - anno_width
            x2n = x1n + w
            y2n = y1n + h

            current_boxes.append([x1n,y1n,x2n,y2n])

        current_image = np.array(output_img)
        current_boxes = np.array(current_boxes,dtype=np.float32)

        return current_image.astype(np.float32), current_boxes, labels


def test_plot_boxes(img, boxes, savedir=None, savename=None):
    img = img.astype(np.uint8)
    img = Image.fromarray(img, 'RGB')
    draw = ImageDraw.Draw(img)

    for ii in range(len(boxes)):
        box = boxes[ii]
        x1,y1,x2,y2 = box
        rgb = (255, 0, 0)
        draw.rectangle([x1, y1, x2, y2], outline=rgb)
    if savename and savedir:
        print("save plot results to %s/%s" % (savedir,savename))
        img.save(f"{savedir}/{savename}")
    return img

