import os

import imageio
import numpy as np
import torch
import torch.utils.data

import utils
from dataloader import train_data_loaders
from models.segmentation import SegmentationModel


def crf_inference(img, probs, t=10, scale_factor=1, labels=21):
    import pydensecrf.densecrf as dcrf
    from pydensecrf.utils import unary_from_softmax

    h, w = img.shape[:2]
    n_labels = labels

    d = dcrf.DenseCRF2D(w, h, n_labels)

    unary = unary_from_softmax(probs)
    unary = np.ascontiguousarray(unary)

    d.setUnaryEnergy(unary)
    d.addPairwiseGaussian(sxy=3 / scale_factor, compat=3)
    d.addPairwiseBilateral(
        sxy=80 / scale_factor, srgb=13, rgbim=np.copy(img), compat=10
    )
    Q = d.inference(t)

    return np.array(Q).reshape((n_labels, h, w))


def main(args):
    if args.output_dir:
        utils.mkdir(args.output_dir)
    print(args)

    device = torch.device(args.device)

    _, data_loader_val = train_data_loaders(
        args.root,
        args.train_list,
        args.train_list,
        args.cam_dir,
        batch_size=args.batch_size,
    )

    model = SegmentationModel()
    model.to(device)
    model.load_state_dict(
        torch.load("./results/model_16.pth")["model"], strict=True,
    )
    model.eval()
    for d in data_loader_val:
        image = d["img"].to(device)
        name = d["name"][0]
        erased_cam_label = d["erased_cam_label"].float().to(device)
        path = os.path.join("outputs", name + ".png")
        img = data_loader_val.dataset._read_image(name)
        with torch.inference_mode():
            output = model(image, erased_cam_label)
            logits = crf_inference(
                np.asarray(img), output[0].detach().cpu().numpy()
            )
            logits = np.argmax(logits, axis=0)
            imageio.imwrite(
                path, logits.astype(np.uint8),
            )


def get_args_parser(add_help=True):
    import argparse

    parser = argparse.ArgumentParser(
        description="PyTorch Segmentation Training", add_help=add_help
    )

    parser.add_argument(
        "--model", default="fcn_resnet101", type=str, help="model name"
    )
    parser.add_argument(
        "--device",
        default="cuda",
        type=str,
        help="device (Use cuda or cpu Default: cuda)",
    )
    parser.add_argument(
        "-b",
        "--batch-size",
        default=8,
        type=int,
        help="images per gpu, the total batch size is $NGPU x batch_size",
    )
    parser.add_argument(
        "--epochs",
        default=30,
        type=int,
        metavar="N",
        help="number of total epochs to run",
    )

    parser.add_argument(
        "-j",
        "--workers",
        default=16,
        type=int,
        metavar="N",
        help="number of data loading workers (default: 16)",
    )
    parser.add_argument(
        "--lr", default=0.01, type=float, help="initial learning rate"
    )
    parser.add_argument(
        "--momentum", default=0.9, type=float, metavar="M", help="momentum"
    )
    parser.add_argument(
        "--wd",
        "--weight-decay",
        default=1e-4,
        type=float,
        metavar="W",
        help="weight decay (default: 1e-4)",
        dest="weight_decay",
    )
    parser.add_argument(
        "--lr-warmup-epochs",
        default=0,
        type=int,
        help="the number of epochs to warmup (default: 0)",
    )
    parser.add_argument(
        "--lr-warmup-method",
        default="linear",
        type=str,
        help="the warmup method (default: linear)",
    )
    parser.add_argument(
        "--lr-warmup-decay", default=0.01, type=float, help="the decay for lr"
    )
    parser.add_argument(
        "--print-freq", default=10, type=int, help="print frequency"
    )
    parser.add_argument(
        "--output-dir", default=".", type=str, help="path to save outputs"
    )
    parser.add_argument(
        "--start-epoch", default=0, type=int, metavar="N", help="start epoch"
    )

    # Mixed precision training parameters
    parser.add_argument(
        "--amp",
        action="store_true",
        help="Use torch.cuda.amp for mixed precision training",
    )

    parser.add_argument(
        "--root", default="../vision/data/raw/VOCdevkit/VOC2012", type=str
    )
    parser.add_argument("--train_list", default="./voc12/train.txt", type=str)
    parser.add_argument("--val_list", default="./voc12/val.txt", type=str)
    parser.add_argument(
        "--cam_dir",
        default="../end-to-end-wsss/outputs/end-to-end-wsss-pipeline/resnet101-deeplabv3/2022-01-16/pseudo_labels/segmentation",
        type=str,
    )

    return parser


if __name__ == "__main__":
    args = get_args_parser().parse_args()
    main(args)
