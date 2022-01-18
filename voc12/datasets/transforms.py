import random

import numpy as np
import torch
from torchvision import transforms as T
from torchvision.transforms import functional as F


def pad_if_smaller(img, size, fill=0):
    min_size = min(img.size)
    if min_size < size:
        ow, oh = img.size
        padh = size - oh if oh < size else 0
        padw = size - ow if ow < size else 0
        img = F.pad(img, (0, 0, padw, padh), fill=fill)
    return img


class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target1=None, target2=None):
        for t in self.transforms:
            image, target1, target2 = t(image, target1, target2)
        return image, target1, target2


class RandomResize:
    def __init__(self, min_size, max_size=None):
        self.min_size = min_size
        if max_size is None:
            max_size = min_size
        self.max_size = max_size

    def __call__(self, image, target1=None, target2=None):
        size = random.randint(self.min_size, self.max_size)
        image = F.resize(image, size)
        if target1 is not None:
            target1 = F.resize(target1, size, interpolation=T.InterpolationMode.NEAREST)
        if target2 is not None:
            target2 = F.resize(target2, size, interpolation=T.InterpolationMode.NEAREST)
        return image, target1, target2


class RandomHorizontalFlip:
    def __init__(self, flip_prob):
        self.flip_prob = flip_prob

    def __call__(self, image, target1=None, target2=None):
        if random.random() < self.flip_prob:
            image = F.hflip(image)
            if target1 is not None:
                target1 = F.hflip(target1)
            if target2 is not None:
                target2 = F.hflip(target2)
        return image, target1, target2


class RandomCrop:
    def __init__(self, size):
        self.size = size

    def __call__(self, image, target1=None, target2=None):
        image = pad_if_smaller(image, self.size)
        crop_params = T.RandomCrop.get_params(image, (self.size, self.size))
        image = F.crop(image, *crop_params)
        if target1 is not None:
            target1 = pad_if_smaller(target1, self.size, fill=255)
            target1 = F.crop(target1, *crop_params)
        if target2 is not None:
            target2 = pad_if_smaller(target2, self.size, fill=255)
            target2 = F.crop(target2, *crop_params)
        return image, target1, target2


class CenterCrop:
    def __init__(self, size):
        self.size = size

    def __call__(self, image, target1=None, target2=None):
        image = F.center_crop(image, self.size)
        if target1 is not None:
            target1 = F.center_crop(target1, self.size)
        if target2 is not None:
            target2 = F.center_crop(target2, self.size)
        return image, target1, target2


class PILToTensor:
    def __call__(self, image, target1=None, target2=None):
        image = F.pil_to_tensor(image)
        if target1 is not None:
            target1 = torch.as_tensor(np.array(target1), dtype=torch.int64)
        if target2 is not None:
            target2 = torch.as_tensor(np.array(target2), dtype=torch.int64)
        return image, target1, target2


class ConvertImageDtype:
    def __init__(self, dtype):
        self.dtype = dtype

    def __call__(self, image, target1=None, target2=None):
        image = F.convert_image_dtype(image, self.dtype)
        return image, target1, target2


class Normalize:
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, image, target1=None, target2=None):
        image = F.normalize(image, mean=self.mean, std=self.std)
        return image, target1, target2
