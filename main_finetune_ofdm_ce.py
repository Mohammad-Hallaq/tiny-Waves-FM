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
import random

import torch
import torch.backends.cudnn as cudnn
from torch.nn import MSELoss, L1Loss
from torch.utils.tensorboard import SummaryWriter

import util.lr_decay as lrd
import util.misc as misc
from util.misc import NativeScalerWithGradNormCount as NativeScaler
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler

import models_ofdm_ce
import math

from engine_finetune_regression_ce import train_one_epoch, evaluate
from dataset_classes.ofdm_channel_estimation import OfdmChannelEstimation
from snr_weighted_loss import WeightedLoss


def get_args_parser():
    parser = argparse.ArgumentParser('MAE fine-tuning for MIMO/OFDM Channel Estimation', add_help=False)
    parser.add_argument('--batch_size', default=64, type=int,
                        help='Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus')
    parser.add_argument('--epochs', default=50, type=int)
    parser.add_argument('--accum_iter', default=1, type=int,
                        help='Accumulate gradient iterations '
                             '(for increasing the effective batch size under memory constraints)')
    # Model parameters
    parser.add_argument('--model', default='ce_small_patch16', type=str, metavar='MODEL',
                        help='Name of model to train')
    parser.add_argument('--input_size', default=224, type=int,
                        help='images input size')
    parser.add_argument('--drop_path', type=float, default=0.1, metavar='PCT',
                        help='Drop path rate (default: 0.1)')
    parser.add_argument('--frozen_blocks', type=int, help='number of encoder blocks to freeze. Freezes all by default')
    parser.add_argument('--snr_token', default=False, action='store_true', help='Whether to use SNR token')
    parser.add_argument('--normalize_labels', action='store_true', default=False,
                        help='normalize labels before training')
    # Optimizer parameters
    parser.add_argument('--clip_grad', type=float, default=None, metavar='NORM',
                        help='Clip gradient norm (default: None, no clipping)')
    parser.add_argument('--weight_decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')

    parser.add_argument('--lr', type=float, default=None, metavar='LR',
                        help='learning rate (absolute lr)')
    parser.add_argument('--blr', type=float, default=1e-3, metavar='LR',
                        help='base learning rate: absolute_lr = base_lr * total_batch_size / 256')
    parser.add_argument('--layer_decay', type=float, default=0.75,
                        help='layer-wise lr decay from ELECTRA/BEiT')

    parser.add_argument('--min_lr', type=float, default=1e-6, metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0')

    parser.add_argument('--warmup_epochs', type=int, default=5, metavar='N',
                        help='epochs to warmup LR')
    parser.add_argument('--loss', default='mse', type=str)
    parser.add_argument('--weighted_loss', action='store_true', default=False, help='use weighted loss')
    parser.add_argument('--loss_mode', default='linear', help='weighted loss mode')

    # * Finetuning params
    parser.add_argument('--finetune', default='',
                        help='finetune from checkpoint')

    # Dataset parameters
    parser.add_argument('--data_path', default='', type=str,
                        help='dataset path')

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
    parser.add_argument('--eval', action='store_true',
                        help='Perform evaluation only')
    parser.add_argument('--dist_eval', action='store_true', default=False,
                        help='Enabling distributed evaluation (recommended during training for faster monitor')
    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=True)

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help=argparse.SUPPRESS)
    parser.add_argument('--local_rank', default=-1, type=int,
                        help=argparse.SUPPRESS)
    parser.add_argument('--dist_on_itp', action='store_true',
                        help=argparse.SUPPRESS)
    parser.add_argument('--dist_url', default='env://',
                        help=argparse.SUPPRESS)

    return parser


def main(args):
    misc.init_distributed_mode(args)

    print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(', ', ',\n'))

    device = torch.device(args.device)

    # fix the seed for reproducibility
    print(f"seed is {args.seed}")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    cudnn.benchmark = True

    dataset_train = OfdmChannelEstimation(os.path.join(Path(args.data_path), 'train_preprocessed'),
                                          normalize_labels=args.normalize_labels)
    dataset_val = OfdmChannelEstimation(Path(args.data_path, 'val_preprocessed'),
                                        normalize_labels=args.normalize_labels)

    sampler_train = RandomSampler(dataset_train)
    sampler_val = SequentialSampler(dataset_val)

    os.makedirs(args.log_dir, exist_ok=True)
    log_writer = SummaryWriter(log_dir=args.log_dir)

    data_loader_train = DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
    )

    data_loader_val = DataLoader(
        dataset_val, sampler=sampler_val,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False
    )

    model = models_ofdm_ce.__dict__[args.model](snr_token=args.snr_token)

    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu', weights_only=False)
        print("Load pre-trained checkpoint from: %s" % args.resume)
        checkpoint_model = checkpoint['model']
        msg = model.load_state_dict(checkpoint_model, strict=True)
        print(msg)

    elif args.finetune:
        checkpoint = torch.load(args.finetune, map_location='cpu', weights_only=False)
        print("Load pre-trained checkpoint from: %s" % args.finetune)
        checkpoint_model = checkpoint['model']
        state_dict = model.state_dict()
        for k in ['head.weight', 'head.bias', 'pos_embed', 'decoder_pred.weight', 'decoder_pred.bias',
                  'patch_embed.proj.weight', 'patch_embed.proj.bias']:
            if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
                print(f"Removing key {k} from pretrained checkpoint")
                del checkpoint_model[k]
        # load pre-trained model
        msg = model.load_state_dict(checkpoint_model, strict=False)
        print(msg)

    if args.frozen_blocks is not None:
        model.freeze_encoder(args.frozen_blocks)
    else:
        model.freeze_encoder()

    model.unfreeze_patch_embed()
    model = model.to(device)
    model_without_ddp = model
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print("Model = %s" % str(model_without_ddp))
    print('number of params (M): %.2f' % (n_parameters / 1.e6))

     # Confirm that only the classification head is trainable
    for name, param in model.named_parameters():
        print(f"{name}: {'Trainable' if param.requires_grad else 'Frozen'}")

    print("The number of frozen blocks is: ", args.frozen_blocks)


    eff_batch_size = args.batch_size * args.accum_iter
    
    if args.lr is None:  # only base_lr is specified
        args.lr = args.blr * eff_batch_size / 256

    print("base lr: %.2e" % (args.lr * 256 / eff_batch_size))
    print("actual lr: %.2e" % args.lr)

    print("accumulate grad iterations: %d" % args.accum_iter)
    print("effective batch size: %d" % eff_batch_size)

    # build optimizer with layer-wise lr decay (lrd)
    param_groups = lrd.param_groups_lrd(model_without_ddp, args.weight_decay, layer_decay=args.layer_decay)
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr)
    loss_scaler = NativeScaler()

    if args.weighted_loss:
        criterion = WeightedLoss(args.loss, args.loss_mode)
    elif args.loss == 'mae':
        criterion = L1Loss()
    else:
        criterion = MSELoss()

    print("criterion = %s" % str(criterion))

    misc.load_model(args=args, model_without_ddp=model_without_ddp, optimizer=optimizer, loss_scaler=loss_scaler)

    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    min_error = math.inf
    for epoch in range(args.start_epoch, args.epochs):
        train_stats = train_one_epoch(
            model, criterion, data_loader_train,
            optimizer, device, epoch, loss_scaler,
            args.clip_grad, None,
            log_writer=log_writer,
            args=args
        )
        if args.output_dir and (epoch + 1 == args.epochs):
            misc.save_model(
                args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                loss_scaler=loss_scaler, epoch=epoch)

        test_stats = evaluate(data_loader_val, model, criterion, device)
        print(f"Error of the network on the {len(dataset_val)} test images: {test_stats['loss']:.4f}")
        min_error = min(min_error, test_stats["loss"])
        print(f'Test error: {min_error:.4f}')

        if test_stats["loss"] == min_error:
            print("A new better model has been saved ... ")
            torch.save(model, os.path.join(args.output_dir, "best_model.pth"))

        if log_writer is not None:
            log_writer.add_scalar('perf/test_loss', test_stats['loss'], epoch)

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     **{f'test_{k}': v for k, v in test_stats.items()},
                     'epoch': epoch,
                     'n_parameters': n_parameters}

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