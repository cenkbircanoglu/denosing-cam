import datetime
import os
import time

import torch
import torch.utils.data
from torch.nn import MSELoss

import utils
from dataloader import train_data_loaders
from models.segmentation import SegmentationModel

mse = MSELoss()


def criterion(inputs, target):
    return mse(inputs, target)


def evaluate(model, data_loader, device, num_classes):
    model.eval()
    confmat = utils.ConfusionMatrix(num_classes)
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = "Test:"
    with torch.inference_mode():
        for d in metric_logger.log_every(data_loader, 100, header):
            image = d["img"].to(device)
            erased_cam_label = d["erased_cam_label"].float().to(device)
            cam_label = d["cam_label"].float().to(device)
            output = model(image, erased_cam_label)
            confmat.update(cam_label.flatten(), output.argmax(1).flatten())

        confmat.reduce_from_all_processes()

    return confmat


def train_one_epoch(
    model,
    criterion,
    optimizer,
    data_loader,
    lr_scheduler,
    device,
    epoch,
    print_freq,
    scaler=None,
):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter(
        "lr", utils.SmoothedValue(window_size=1, fmt="{value}")
    )
    header = f"Epoch: [{epoch}]"
    for d in metric_logger.log_every(data_loader, print_freq, header):
        image = d["img"].to(device)
        erased_cam_label = d["erased_cam_label"].float().to(device)
        cam_label = d["cam_label"].float().to(device)
        with torch.cuda.amp.autocast(enabled=scaler is not None):
            output = model(image, erased_cam_label)
            loss = criterion(output, cam_label)

        optimizer.zero_grad()
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        lr_scheduler.step()

        metric_logger.update(
            loss=loss.item(), lr=optimizer.param_groups[0]["lr"]
        )


def main(args):
    if args.output_dir:
        utils.mkdir(args.output_dir)
    print(args)

    device = torch.device(args.device)

    data_loader_train, data_loader_val = train_data_loaders(
        args.root,
        args.train_list,
        args.val_list,
        args.cam_dir,
        batch_size=args.batch_size,
    )

    model = SegmentationModel()
    model.to(device)

    model_without_ddp = model

    params_to_optimize = [
        {
            "params": [
                p
                for p in model_without_ddp.backbone.parameters()
                if p.requires_grad
            ]
        },
        {
            "params": [
                p
                for p in model_without_ddp.classifier.parameters()
                if p.requires_grad
            ]
        },
    ]
    optimizer = torch.optim.SGD(
        params_to_optimize,
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
    )

    scaler = torch.cuda.amp.GradScaler() if args.amp else None

    iters_per_epoch = len(data_loader_train)
    main_lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda x: (
            1 - x / (iters_per_epoch * (args.epochs - args.lr_warmup_epochs))
        )
        ** 0.9,
    )

    if args.lr_warmup_epochs > 0:
        warmup_iters = iters_per_epoch * args.lr_warmup_epochs
        args.lr_warmup_method = args.lr_warmup_method.lower()
        if args.lr_warmup_method == "linear":
            warmup_lr_scheduler = torch.optim.lr_scheduler.LinearLR(
                optimizer,
                start_factor=args.lr_warmup_decay,
                total_iters=warmup_iters,
            )
        elif args.lr_warmup_method == "constant":
            warmup_lr_scheduler = torch.optim.lr_scheduler.ConstantLR(
                optimizer, factor=args.lr_warmup_decay, total_iters=warmup_iters
            )
        else:
            raise RuntimeError(
                f"Invalid warmup lr method '{args.lr_warmup_method}'. Only linear and constant are supported."
            )
        lr_scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[warmup_lr_scheduler, main_lr_scheduler],
            milestones=[warmup_iters],
        )
    else:
        lr_scheduler = main_lr_scheduler

    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        train_one_epoch(
            model,
            criterion,
            optimizer,
            data_loader_train,
            lr_scheduler,
            device,
            epoch,
            args.print_freq,
            scaler,
        )
        confmat = evaluate(model, data_loader_val, device=device, num_classes=21)
        print(confmat)
        checkpoint = {
            "model": model_without_ddp.state_dict(),
            "optimizer": optimizer.state_dict(),
            "lr_scheduler": lr_scheduler.state_dict(),
            "epoch": epoch,
            "args": args,
        }
        if args.amp:
            checkpoint["scaler"] = scaler.state_dict()
        utils.save_on_master(
            checkpoint, os.path.join(args.output_dir, f"model_{epoch}.pth")
        )
        utils.save_on_master(
            checkpoint, os.path.join(args.output_dir, "checkpoint.pth")
        )

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print(f"Training time {total_time_str}")


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
    parser.add_argument(
        "--train_list", default="./voc12/train_aug.txt", type=str
    )
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
