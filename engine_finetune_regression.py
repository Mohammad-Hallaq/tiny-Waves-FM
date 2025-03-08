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
from typing import Iterable, Optional

import torch

from timm.data import Mixup
from timm.utils import accuracy

import util.misc as misc
import util.lr_sched as lr_sched


def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, max_norm: float = 0,
                    mixup_fn: Optional[Mixup] = None, log_writer=None,
                    args=None):
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 20

    accum_iter = args.accum_iter

    optimizer.zero_grad()

    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))

    for data_iter_step, (samples, targets) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):

        # we use a per iteration (instead of per epoch) lr scheduler
        if data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)

        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        with torch.cuda.amp.autocast():
            outputs = model(samples)
            loss = criterion(outputs, targets)

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        loss /= accum_iter
        loss_scaler(loss, optimizer, clip_grad=max_norm,
                    parameters=model.parameters(), create_graph=False,
                    update_grad=(data_iter_step + 1) % accum_iter == 0)
        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.zero_grad()

        torch.cuda.synchronize()

        metric_logger.update(loss=loss_value)
        min_lr = 10.
        max_lr = 0.
        for group in optimizer.param_groups:
            min_lr = min(min_lr, group["lr"])
            max_lr = max(max_lr, group["lr"])

        metric_logger.update(lr=max_lr)

        loss_value_reduce = misc.all_reduce_mean(loss_value)
        if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
            """ We use epoch_1000x as the x-axis in tensorboard.
            This calibrates different curves when batch size changes.
            """
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            log_writer.add_scalar('loss', loss_value_reduce, epoch_1000x)
            log_writer.add_scalar('lr', max_lr, epoch_1000x)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(data_loader, model, criterion, device, coord_min, coord_max):
    # Ensure coord_min and coord_max are on the same device.
    coord_min = coord_min.to(device)
    coord_max = coord_max.to(device)

    def reverse_normalize(x):
        return (x + 1) / 2 * (coord_max - coord_min) + coord_min

    metric_logger = misc.MetricLogger(delimiter="  ")
    header = 'Test:'

    model.eval()
    distances_list = []

    for batch in metric_logger.log_every(data_loader, 10, header):
        images = batch[0]
        target = batch[-1]
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        # Compute model output and loss
        with torch.cuda.amp.autocast():
            output = model(images)
            loss = criterion(output, target)

        metric_logger.update(loss=loss.item())

        # Compute unnormalized predictions and true positions on GPU.
        pred_position = reverse_normalize(output)
        true_position = reverse_normalize(target)

        # Compute Euclidean distance error for each sample in the batch.
        distance_error = torch.sqrt(torch.sum((pred_position - true_position) ** 2, dim=1))
        distances_list.append(distance_error)

    metric_logger.synchronize_between_processes()

    # Concatenate all error distances and compute mean and standard deviation on GPU.
    all_distances = torch.cat(distances_list, dim=0)
    mean_distance = all_distances.mean().item()
    std_distance = all_distances.std().item()

    print('* loss {losses.global_avg:.3f}  Mean distance error: {mean_distance:.3f}  '
          'Stdev distance error: {std_distance:.3f}'
          .format(losses=metric_logger.loss, mean_distance=mean_distance, std_distance=std_distance))

    results = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    results['mean_distance_error'] = mean_distance
    results['stdev_distance_error'] = std_distance
    return results
