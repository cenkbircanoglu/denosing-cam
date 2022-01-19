import torch

from voc12.datasets import presets
from voc12.datasets.pseudo_seg import VOC12ClsCAMDataset


def train_data_loaders(root, train_list, val_list, cam_dir, batch_size):
    transform_train = presets.PresetTrain(base_size=520, crop_size=512)
    ds_train = VOC12ClsCAMDataset(
        img_name_list_path=train_list,
        root=root,
        transforms=transform_train,
        erase=True,
        cam_dir=cam_dir,
    )
    transform_val = presets.PresetEval()

    ds_val = VOC12ClsCAMDataset(
        img_name_list_path=val_list,
        root=root,
        transforms=transform_val,
        cam_dir=cam_dir,
        erase=False,
    )
    data_loader_train = torch.utils.data.DataLoader(
        ds_train, batch_size=batch_size, num_workers=8, drop_last=True,
    )

    data_loader_val = torch.utils.data.DataLoader(
        ds_val, batch_size=1, num_workers=4,
    )
    return data_loader_train, data_loader_val


def infer_data_loader(cfg_dataset=None, num_workers=2):
    transform_val = presets.PresetEval()

    ds_val = VOC12ClassificationDataset(
        img_name_list_path=cfg_dataset.infer_list,
        root=cfg_dataset.root,
        transforms=transform_val,
        scales=[1.0, 0.5, 1.5, 2.0],
    )

    data_loader_val = torch.utils.data.DataLoader(
        ds_val, batch_size=1, num_workers=num_workers,
    )
    return data_loader_val
