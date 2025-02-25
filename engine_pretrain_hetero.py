# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# BEiT: https://github.com/microsoft/unilm/tree/master/beit
# --------------------------------------------------------
import math
import sys
from typing import Iterable

import torch

import util.misc as misc
import util.lr_sched as lr_sched


def train_one_epoch(model: torch.nn.Module,
                    data_loader_A, data_loader_B, data_loader_C,
                    optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler,
                    log_writer=None,
                    args=None):
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = f'Epoch: [{epoch}]'
    print_freq = 20

    accum_iter = args.accum_iter
    optimizer.zero_grad()

    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))

    # Create iterators for each dataloader
    iter_A = iter(data_loader_A)
    iter_B = iter(data_loader_B)
    iter_C = iter(data_loader_C)

    # Determine total iterations as twice the maximum of the two dataloader lengths.
    total_iterations = 3 * max(len(data_loader_A), len(data_loader_B), len(data_loader_C))

    for data_iter_step in range(total_iterations):
        mod = data_iter_step % 3
        if mod == 0:
            # Use dataset A.
            try:
                samples, _ = next(iter_A)
            except StopIteration:
                iter_A = iter(data_loader_A)
                samples, _ = next(iter_A)
        elif mod == 1:
            # Use dataset B.
            try:
                samples, _ = next(iter_B)
            except StopIteration:
                iter_B = iter(data_loader_B)
                samples, _ = next(iter_B)
        else:  # mod == 2
            # Use dataset C.
            try:
                samples, _ = next(iter_C)
            except StopIteration:
                iter_C = iter(data_loader_C)
                samples, _ = next(iter_C)

        # Optionally print iteration info every print_freq steps
        if data_iter_step % print_freq == 0:
            print(header, f"Iteration: {data_iter_step}/{total_iterations}")

        # Adjust learning rate (per iteration)
        if data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate(optimizer, data_iter_step / total_iterations + epoch, args)

        samples = samples.to(device, non_blocking=True)

        with torch.cuda.amp.autocast():
            loss, _, _ = model(samples, mask_ratio=args.mask_ratio)

        loss_value = loss.item()
        if not math.isfinite(loss_value):
            print(f"Loss is {loss_value}, stopping training")
            sys.exit(1)

        loss /= accum_iter
        loss_scaler(loss, optimizer, parameters=model.parameters(),
                    update_grad=(data_iter_step + 1) % accum_iter == 0)
        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.zero_grad()

        torch.cuda.synchronize()
        metric_logger.update(loss=loss_value)
        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(lr=lr)

        loss_value_reduce = misc.all_reduce_mean(loss_value)
        if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
            # We use epoch_1000x as the x-axis in tensorboard.
            epoch_1000x = int((data_iter_step / total_iterations + epoch) * 1000)
            log_writer.add_scalar('train_loss', loss_value_reduce, epoch_1000x)
            log_writer.add_scalar('lr', lr, epoch_1000x)

    # Gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
