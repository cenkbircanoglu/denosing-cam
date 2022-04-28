import os
from abc import ABC

import PIL
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import transforms as T, InterpolationMode

from voc12.utils import (
    load_img_name_list,
    get_img_path,
    load_image_label_list_from_npy,
)


class VOC12BaseDataset(Dataset, ABC):
    def __init__(self, img_name_list_path: str, root: str, transforms=None):
        self.img_name_list = load_img_name_list(img_name_list_path)
        self.label_list = load_image_label_list_from_npy(self.img_name_list)
        self.root = root
        self.transforms = transforms

    def __len__(self):
        return len(self.img_name_list)

    def _read_image(self, name_str: str) -> PIL.Image:
        return Image.open(get_img_path(name_str, self.root)).convert("RGB")

    def _read_cls_label(self, idx):
        return torch.from_numpy(self.label_list[idx])

    def _read_seg_label(self, name_str):
        path = os.path.join(self.seg_label_dir, name_str + ".png")
        return Image.open(path)

    def _read_aff_label(self, name_str):
        path = os.path.join(self.aff_label_dir, name_str + ".png")
        return Image.open(path)

    def _calculate_aff_labels(self, label):
        assert label.size(0) == label.size(1)
        resized_label = T.Resize(
            size=label.size(0) // 4, interpolation=InterpolationMode.NEAREST
        )(label.unsqueeze(0)).squeeze(0)
        return self.extract_aff_lab_func(resized_label.numpy().astype(np.uint64))

    @staticmethod
    def create_scales(img_tensor, scales=[1.0, 0.5, 1.5, 2.0]):
        img_tensor = torch.stack([img_tensor, img_tensor.flip(-1)])
        height, width = img_tensor.shape[2:]
        img_tensor_list = []
        for scale in scales:
            if scale == 1.0:
                img_tensor_list.append(img_tensor)
            else:
                target_size = (
                    int(round(height * scale)),
                    int(round(width * scale)),
                )
                resized_x = T.Resize(size=target_size)(img_tensor)
                img_tensor_list.append(resized_x)
        return img_tensor_list


if __name__ == "__main__":
    x = torch.rand([3, 224, 224])
    x_list = VOC12BaseDataset.create_scales(x)
    assert len(x_list) == 4
    assert x_list[0].shape == (2, 3, 224, 224)
    assert x_list[1].shape == (2, 3, 112, 112)
    assert x_list[2].shape == (2, 3, 336, 336)
    assert x_list[3].shape == (2, 3, 448, 448)
