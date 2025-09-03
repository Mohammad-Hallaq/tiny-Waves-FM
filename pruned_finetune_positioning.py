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
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import SequentialSampler, DataLoader, RandomSampler

import util.lr_decay as lrd
import util.misc as misc
from util.misc import NativeScalerWithGradNormCount as NativeScaler
from timm.layers import trunc_normal_
from timm.data.mixup import Mixup
from advanced_finetuning.lora import create_lora_model

import models_vit
import math

from pruned_engine_finetune_regression import train_one_epoch, evaluate
from dataset_classes.positioning import Positioning5G
from advanced_finetuning.prefix import create_prefix_tuning_model


def get_args_parser():
    parser = argparse.ArgumentParser('MAE fine-tuning for 5G NR Positioning', add_help=False)
    parser.add_argument('--batch_size', default=256, type=int,
                        help='Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus')
    parser.add_argument('--epochs', default=300, type=int)
    parser.add_argument('--accum_iter', default=1, type=int,
                        help='Accumulate gradient iterations ('
                             'for increasing the effective batch size under memory constraints)')
    parser.add_argument('--seed', type=int, default=42,
                        help='seed for initializing training.')
    # Model parameters
    # parser.add_argument('--model', default='vit_small_patch16', type=str, metavar='MODEL',
    #                     help='Name of model to train')
    
    parser.add_argument('--model_path', default='', type=str, metavar='MODEL',
                        help='Path to the pruned model')

    parser.add_argument('--input_size', default=224, type=int,
                        help='images input size')
    # parser.add_argument('--head_layers', default=1, type=int,
    #                     help='number of layers in task head')
    parser.add_argument('--drop_path', type=float, default=0.1, metavar='PCT',
                        help='Drop path rate (default: 0.1)')
    # parser.add_argument('--frozen_blocks', type=int, help='number of encoder blocks to freeze. Freezes all by default')
    parser.add_argument('--tanh', action='store_true', default=False)

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

    parser.add_argument('--warmup_epochs', type=int, default=15, metavar='N',
                        help='epochs to warmup LR')

    # LoRa Parameters
    parser.add_argument('--lora', action='store_true', help='Whether to use LoRa (default: False)')

    parser.add_argument('--lora_rank', type=int, default=8, help='Rank of LoRa (default: 8)')

    parser.add_argument('--lora_alpha', type=float, default=1, help='Alpha for LoRa (default: 0.5)')

    # Prefix Tuning Parameters
    parser.add_argument('--prefix_tuning', action='store_true', help='Whether to use prefix tuning')

    parser.add_argument('--num_prefix_tokens', type=int, default=20, help='Number of prefix tokens')

    # * Finetuning params
    parser.add_argument('--finetune', default='',
                        help='finetune from checkpoint')
    parser.add_argument('--global_pool', default='avg')

    # Dataset parameters
    parser.add_argument('--scene', default='outdoor', type=str, choices=['indoor', 'outdoor'],
                        help='Scene to use')
    parser.add_argument('--data_path', default='', type=str,
                        help='dataset path')
    parser.add_argument('--nb_outputs', default=3, type=int,
                        help='number of outputs')

    parser.add_argument('--output_dir', default='./pruning_results/positioning',
                        help='path where to save, empty for no saving')
    parser.add_argument('--log_dir', default='./pruning_results/positioning',
                        help='path where to tensorboard log')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
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
    parser.add_argument('--world_size', default=1, type=int,
                        help=argparse.SUPPRESS)
    parser.add_argument('--local_rank', default=-1, type=int, help=argparse.SUPPRESS)
    parser.add_argument('--dist_on_itp', action='store_true', help=argparse.SUPPRESS)
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

    dataset_train = Positioning5G(Path(os.path.join(args.data_path, f'{args.scene}/train')))
    dataset_val = Positioning5G(Path(os.path.join(args.data_path, f'{args.scene}/test')))

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
    print("The path of the pruned model is: ", args.model_path)
    model = torch.load(args.model_path, weights_only=False)

    print(model)

    # if args.lora:
    #     model = create_lora_model(model, args.lora_rank, args.lora_alpha)
    # elif args.prefix_tuning:
    #     model = create_prefix_tuning_model(model, pool='token')

    # elif args.finetune:
    #     checkpoint = torch.load(args.finetune, map_location='cpu', weights_only=False)
    #     print("Load pre-trained checkpoint from: %s" % args.finetune)
    #     checkpoint_model = checkpoint['model']
    #     state_dict = model.state_dict()
    #     for k in ['head.weight', 'head.bias', 'pos_embed', 'patch_embed.proj.weight', 'patch_embed.proj.bias']:
    #         if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
    #             print(f"Removing key {k} from pretrained checkpoint")
    #             del checkpoint_model[k]
    #     # load pre-trained model
    #     msg = model.load_state_dict(checkpoint_model, strict=False)
    #     print(msg)
    #     # manually initialize fc layer
    #     if args.head_layers == 1:
    #         trunc_normal_(model.head.weight, std=2e-5)

    # if args.lora:
    #     model.freeze_encoder_lora()
    # elif args.prefix_tuning:
    #     model.freeze_encoder_prefix()
    # elif args.frozen_blocks is not None:
    #     model.freeze_encoder(args.frozen_blocks)
    # else:
    #     model.freeze_encoder()

    # model.unfreeze_patch_embed()
    for param in model.blocks.parameters():
            param.requires_grad = False
    
    # Freeze positional embeddings and tokens
    if hasattr(model, "cls_token"):
        model.cls_token.requires_grad = False
    if hasattr(model, "pos_embed"):
        model.pos_embed.requires_grad = False
    if hasattr(model, "mask_token"):
        model.mask_token.requires_grad = False
    if hasattr(model, "decoder_pos_embed"):
        model.decoder_pos_embed.requires_grad = False

    # Confirm that only the classification head is trainable
    for name, param in model.named_parameters():
        print(f"{name}: {'Trainable' if param.requires_grad else 'Frozen'}")

    model = model.to(device)
    model_without_ddp = model
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print("Model = %s" % str(model_without_ddp))
    print('number of params (M): %.2f' % (n_parameters / 1.e6))

    # Confirm that only the classification head is trainable
    for name, param in model.named_parameters():
        print(f"{name}: {'Trainable' if param.requires_grad else 'Frozen'}")

    # print("The number of frozen blocks is: ", args.frozen_blocks)

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

    criterion = torch.nn.MSELoss()
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
        if args.output_dir and (epoch % 10 == 0 or (epoch + 1) == args.epochs):
            misc.save_model(
                args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                loss_scaler=loss_scaler, epoch=epoch)

        test_stats = evaluate(
            data_loader_val, model, criterion, device,
            dataset_train.coord_nominal_min, dataset_train.coord_nominal_max
        )
        print(f"Test loss on the {len(dataset_val)} test images: {test_stats['loss']:.4f}")
        print(f"Mean distance error: {test_stats['mean_distance_error']:.4f}, "
              f"Stdev distance error: {test_stats['stdev_distance_error']:.4f}")

        # Use the mean distance error as the main error metric for updating min_error.
        min_error = min(min_error, test_stats["mean_distance_error"])
        print(f'Minimum mean distance error: {min_error:.4f}')

        if test_stats["mean_distance_error"] == min_error:
            print("A new better model has been saved ... ")
            torch.save(model, os.path.join(args.output_dir, "best_model.pth"))

        if log_writer is not None:
            log_writer.add_scalar('perf/test_loss', test_stats['loss'], epoch)
            log_writer.add_scalar('perf/test_mean_distance_error', test_stats['mean_distance_error'], epoch)
            log_writer.add_scalar('perf/test_stdev_distance_error', test_stats['stdev_distance_error'], epoch)

        if args.lora:
            log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                         **{f'test_{k}': v for k, v in test_stats.items()},
                         'epoch': epoch,
                         'n_parameters': n_parameters,
                         'lora_rank': args.lora_rank,
                         'lora_alpha': args.lora_alpha}
        elif args.prefix_tuning:
            log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                         **{f'test_{k}': v for k, v in test_stats.items()},
                         'epoch': epoch,
                         'n_parameters': n_parameters,
                         'num_prefix_tokens': args.num_prefix_tokens}
        else:
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
