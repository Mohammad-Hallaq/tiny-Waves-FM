# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# BEiT: https://github.com/microsoft/unilm/tree/master/beit
# --------------------------------------------------------
import argparse
import datetime
import json
import numpy as np
import os
import time
from pathlib import Path

import torch
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from gaussian_noise import AddGaussianNoise
from dataset_classes.pretrain_csi_5g import CSI5G
from dataset_classes.pretrain_csi_wifi import CSIWiFi
from dataset_classes.spectrogram_images import SpectrogramImages

from torch.utils.data import DataLoader, RandomSampler
import timm.optim.optim_factory as optim_factory

import util.misc as misc
from util.misc import NativeScalerWithGradNormCount as NativeScaler

import models_mae_hetero
from engine_pretrain_hetero import train_one_epoch


def get_args_parser():
    parser = argparse.ArgumentParser('MAE pre-training on all datasets', add_help=False)
    parser.add_argument('--batch_size', default=64, type=int,
                        help='Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus')
    parser.add_argument('--epochs', default=400, type=int)
    parser.add_argument('--accum_iter', default=1, type=int,
                        help='Accumulate gradient iterations '
                             '(for increasing the effective batch size under memory constraints)')

    # Model parameters
    parser.add_argument('--model', default='mae_vit_small_patch16', type=str, metavar='MODEL',
                        help='Name of model to train')

    parser.add_argument('--input_size', default=224, type=int,
                        help='images input size')

    parser.add_argument('--mask_ratio', default=0.75, type=float,
                        help='Masking ratio (percentage of removed patches).')

    # Optimizer parameters
    parser.add_argument('--weight_decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')

    parser.add_argument('--lr', type=float, default=None, metavar='LR',
                        help='learning rate (absolute lr)')
    parser.add_argument('--blr', type=float, default=1e-3, metavar='LR',
                        help='base learning rate: absolute_lr = base_lr * total_batch_size / 256')
    parser.add_argument('--min_lr', type=float, default=0., metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0')

    parser.add_argument('--warmup_epochs', type=int, default=40, metavar='N',
                        help='epochs to warmup LR')

    # Dataset parameters
    parser.add_argument('--data_path', default=[], type=str, nargs='+',
                        help='dataset path(s)')
    parser.add_argument('--augmentation', action='store_true', default=False,
                        help='apply data augmentation')
    parser.add_argument('--augment_factor', type=int, default=4,
                        help='factor for increase factor in CSI data')
    parser.add_argument('--csi_subsampling', action='store_true', default=False,
                        help='Use half batch size for CSI data')
    parser.add_argument('--output_dir', default='./output_dir',
                        help='path where to save, empty for no saving')
    parser.add_argument('--log_dir', default='./output_dir',
                        help='path where to tensorboard log')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='',
                        help='resume from checkpoint')

    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=True)

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int, help=argparse.SUPPRESS)
    parser.add_argument('--local_rank', default=-1, type=int, help=argparse.SUPPRESS)
    parser.add_argument('--dist_on_itp', action='store_true', help=argparse.SUPPRESS)
    parser.add_argument('--dist_url', default='env://', help=argparse.SUPPRESS)

    return parser


def main(args):
    misc.init_distributed_mode(args)
    print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(', ', ',\n'))

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)

    cudnn.benchmark = True

    transform_train = transforms.Compose([
        transforms.functional.pil_to_tensor,
        transforms.Lambda(lambda x: 10 * torch.log10(x + 1e-12)),
        transforms.Lambda(lambda x: (x + 120) / (-0.5 + 120)),
        transforms.Resize((224, 224), antialias=True,
                          interpolation=InterpolationMode.BICUBIC),  # Resize
        transforms.Normalize(mean=[0.451], std=[0.043])  # Normalize
    ])

    dataset_train_one = SpectrogramImages(args.data_path[:-1], transform=transform_train)

    augment_transforms = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        AddGaussianNoise(mean=0.0, std=0.05)]
    )
    if args.augmentation:
        dataset_train_two = CSIWiFi(args.data_path[-1], augment_transforms=augment_transforms, factor=args.augment_factor)
    else:
        dataset_train_two = CSIWiFi(args.data_path[-1])

    print(dataset_train_one, dataset_train_two)

    sampler_train_one = RandomSampler(dataset_train_one)
    sampler_train_two = RandomSampler(dataset_train_two)

    os.makedirs(args.log_dir, exist_ok=True)
    log_writer = SummaryWriter(log_dir=args.log_dir)

    data_loader_train_one = DataLoader(
        dataset_train_one, sampler=sampler_train_one,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True)

    if args.csi_subsampling:
        data_loader_train_two = DataLoader(
            dataset_train_two, sampler=sampler_train_two,
            batch_size=args.batch_size // 2,
            num_workers=args.num_workers,
            pin_memory=args.pin_mem,
            drop_last=True)
    else:
        data_loader_train_two = DataLoader(
            dataset_train_two, sampler=sampler_train_two,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=args.pin_mem,
            drop_last=True)

    # define the model
    model = models_mae_hetero.__dict__[args.model](norm_pix_loss=False, in_chans=[1, 3])
    model.to(device)

    model_without_ddp = model
    print("Model = %s" % str(model_without_ddp))

    eff_batch_size = args.batch_size * args.accum_iter
    
    if args.lr is None:  # only base_lr is specified
        args.lr = args.blr * eff_batch_size / 256

    print("base lr: %.2e" % (args.lr * 256 / eff_batch_size))
    print("actual lr: %.2e" % args.lr)

    print("accumulate grad iterations: %d" % args.accum_iter)
    print("effective batch size: %d" % eff_batch_size)

    # following timm: set wd as 0 for bias and norm layers
    param_groups = optim_factory.param_groups_weight_decay(model_without_ddp, args.weight_decay)
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr, betas=(0.9, 0.95))
    print(optimizer)
    loss_scaler = NativeScaler()

    misc.load_model(args=args, model_without_ddp=model_without_ddp, optimizer=optimizer, loss_scaler=loss_scaler)

    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        train_stats = train_one_epoch(
            model, [data_loader_train_one, data_loader_train_two],
            optimizer, device, epoch, loss_scaler,
            log_writer=log_writer,
            args=args
        )

        if args.output_dir and (epoch % 20 == 0 or epoch + 1 == args.epochs):
            misc.save_model(
                args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                loss_scaler=loss_scaler, epoch=epoch)

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()}, 'epoch': epoch}

        if args.output_dir:
            if log_writer is not None:
                log_writer.flush()
            with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + "\n")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
