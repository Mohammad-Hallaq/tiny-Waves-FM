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

import torch

import util.misc as misc
import util.lr_sched as lr_sched


class RoundRobinLoader:
    def __init__(self, loaders):
        self.loaders = loaders
        # Compute total iterations from the lengths of all loaders.
        self.total = sum(len(loader) for loader in loaders.values())

    def __iter__(self):
        # Create iterators for each dataloader.
        iterators = {k: iter(loader) for k, loader in self.loaders.items()}
        order = list(self.loaders.keys())
        # Continue until all iterators are exhausted.
        while order:
            for key in order.copy():
                try:
                    samples, labels = next(iterators[key])
                    yield key, samples, labels
                except StopIteration:
                    order.remove(key)

    def __len__(self):
        return self.total


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

    # Compute total iterations as the sum of batches in each dataloader.
    total_iterations = len(data_loader_A) + len(data_loader_B) + len(data_loader_C)

    # Create a dictionary of your dataloaders.
    loaders = {
        'A': data_loader_A,
        'B': data_loader_B,
        'C': data_loader_C
    }

    # Create the round-robin iterator.
    combined_iter = RoundRobinLoader(loaders)

    for data_iter_step, (key, samples, labels) in enumerate(metric_logger.log_every(combined_iter, print_freq, header)):

        # Adjust learning rate (per iteration).
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
            # Use epoch_1000x as the x-axis in tensorboard.
            epoch_1000x = int((data_iter_step / total_iterations + epoch) * 1000)
            log_writer.add_scalar('train_loss', loss_value_reduce, epoch_1000x)
            log_writer.add_scalar('lr', lr, epoch_1000x)

    # After the loop, synchronize and print the averaged stats.
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}