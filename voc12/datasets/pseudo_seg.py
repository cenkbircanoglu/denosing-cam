import os

import numpy as np
from PIL import Image
from torchvision import transforms

from voc12.datasets.base import VOC12BaseDataset
from voc12.utils import decode_int_filename


class VOC12ClsCAMDataset(VOC12BaseDataset):
    def __init__(
        self,
        img_name_list_path,
        root,
        transforms=None,
        cam_dir=None,
        erase=False,
    ):
        super().__init__(img_name_list_path, root, transforms)
        self.cam_dir = cam_dir
        self.erase = erase

    def _read_cam_label(self, name_str):
        path = os.path.join(self.cam_dir, name_str + ".png")
        if os.path.exists(path):
            return Image.open(path)
        img = self._cached_img
        return Image.fromarray(
            np.zeros((img.size[1], img.size[0])).astype(np.uint8)
        )

    def __getitem__(self, idx):
        name = self.img_name_list[idx]
        name_str = decode_int_filename(name)

        img = self._read_image(name_str)
        self._cached_img = img
        cls_label = self._read_cls_label(idx)
        cam_label = self._read_cam_label(name_str)

        if self.transforms is not None:
            img, cam_label, _ = self.transforms(img, cam_label)
        cam_label = cam_label.unsqueeze(0)
        erased_cam_label = cam_label
        if self.erase:
            erased_cam_label = transforms.RandomErasing()(cam_label)
        from torch.nn.functional import one_hot

        cam_label = one_hot(cam_label, 21).transpose(1, 3)
        d = {
            "img": img,
            "cls_label": cls_label,
            "idx": idx,
            "name": name_str,
            "cam_label": cam_label.squeeze(0),
            "erased_cam_label": erased_cam_label,
        }
        return d
