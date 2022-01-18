import torch

import voc12.datasets.transforms as T


class PresetTrain:
    def __init__(
            self,
            base_size,
            crop_size,
            hflip_prob=0.5,
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
    ):
        min_size = int(0.4 * base_size)
        max_size = int(2.0 * base_size)

        trans = [T.RandomResize(min_size, max_size)]
        if hflip_prob > 0:
            trans.append(T.RandomHorizontalFlip(hflip_prob))
        trans.extend(
            [
                T.RandomCrop(crop_size),
                T.PILToTensor(),
                T.ConvertImageDtype(torch.float),
                T.Normalize(mean=mean, std=std),
            ]
        )
        self.transforms = T.Compose(trans)

    def __call__(self, img, target1=None, target2=None):
        return self.transforms(img, target1, target2)


class PresetEval:
    def __init__(self, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        self.transforms = T.Compose(
            [
                T.PILToTensor(),
                T.ConvertImageDtype(torch.float),
                T.Normalize(mean=mean, std=std),
            ]
        )

    def __call__(self, img, target1=None, target2=None):
        return self.transforms(img, target1, target2)
