import os.path
import pathlib

import numpy as np

IMG_FOLDER_NAME = "JPEGImages"
ANNOT_FOLDER_NAME = "Annotations"
IGNORE = 255

CAT_LIST = [
    "aeroplane",
    "bicycle",
    "bird",
    "boat",
    "bottle",
    "bus",
    "car",
    "cat",
    "chair",
    "cow",
    "diningtable",
    "dog",
    "horse",
    "motorbike",
    "person",
    "pottedplant",
    "sheep",
    "sofa",
    "train",
    "tvmonitor",
]

N_CAT = len(CAT_LIST)

CAT_NAME_TO_NUM = dict(zip(CAT_LIST, range(len(CAT_LIST))))

cls_labels_dict = np.load(
    f"{os.path.dirname(pathlib.Path(__file__).absolute())}/cls_labels.npy",
    allow_pickle=True,
).item()


def decode_int_filename(int_filename) -> str:
    s = str(int(int_filename))
    return s[:4] + "_" + s[4:]


def load_image_label_list_from_npy(img_name_list) -> np.array:
    return np.array([cls_labels_dict[img_name] for img_name in img_name_list])


def get_img_path(img_name, voc12_root) -> str:
    if not isinstance(img_name, str):
        img_name = decode_int_filename(img_name)
    return os.path.join(voc12_root, IMG_FOLDER_NAME, img_name + ".jpg")


def load_img_name_list(dataset_path):
    img_name_list = np.loadtxt(dataset_path, dtype=np.int32)

    return img_name_list
