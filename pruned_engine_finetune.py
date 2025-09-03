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
import torch.nn.functional as F

import timm
from timm.data import Mixup
from timm.utils import accuracy

import util.misc as misc
import util.lr_sched as lr_sched

def no_mask_forward(self, imgs, mask_ratio=0.0):
        latent, _, ids_restore = self.forward_encoder(imgs, mask_ratio)
        out = self.forward_decoder(latent, ids_restore)

        # Example: use CLS token only for classification
        cls_output = out.mean(dim=1)  # shape: [B, decoder_pred_dim]  
        return cls_output

def forward(self, x):
    """https://github.com/huggingface/pytorch-image-models/blob/054c763fcaa7d241564439ae05fbe919ed85e614/timm/models/vision_transformer.py#L79"""
    B, N, C = x.shape
    qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    q, k, v = qkv.unbind(0)
    q, k = self.q_norm(q), self.k_norm(k)

    if self.fused_attn:
        x = F.scaled_dot_product_attention(
            q, k, v,
            dropout_p=self.attn_drop.p,
        )
    else:
        q = q * self.scale
        attn = q @ k.transpose(-2, -1)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = attn @ v

    x = x.transpose(1, 2).reshape(B, N, -1) # original implementation: x = x.transpose(1, 2).reshape(B, N, C)
    x = self.proj(x)
    x = self.proj_drop(x)
    return x

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
 
        if mixup_fn is not None:
            samples, targets = mixup_fn(samples, targets)

        for m in model.modules():
            if isinstance(m, timm.models.vision_transformer.Attention):
                m.forward = forward.__get__(m, timm.models.vision_transformer.Attention) 


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
def evaluate(data_loader, model, criterion, device):
    metric_logger = misc.MetricLogger(delimiter="  ")
    header = 'Test:'

    # switch to evaluation mode
    model.eval()

    per_class_correct = None
    per_class_total = None

    for batch in metric_logger.log_every(data_loader, 10, header):
        images = batch[0]
        target = batch[-1]
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        # compute output
        with torch.cuda.amp.autocast():
            output = model(images)
            loss = criterion(output, target)

        # compute top-1 and top-3 accuracy for the batch
        acc1, acc3 = accuracy(output, target, topk=(1, 3))

        batch_size = images.shape[0]
        metric_logger.update(loss=loss.item())
        metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
        metric_logger.meters['acc3'].update(acc3.item(), n=batch_size)

        # Initialize per-class tracking on the first batch
        if per_class_correct is None:
            num_classes = output.shape[1]
            per_class_correct = torch.zeros(num_classes, device=device)
            per_class_total = torch.zeros(num_classes, device=device)

        # Get top-1 predictions
        _, pred = output.max(1)
        # Update per-class counts for each sample in the batch
        for i in range(batch_size):
            label = target[i]
            per_class_total[label] += 1
            if pred[i] == label:
                per_class_correct[label] += 1

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()

    # Compute per-class accuracy (avoiding division by zero)
    per_class_acc = torch.where(
        per_class_total > 0,
        per_class_correct / per_class_total * 100,
        torch.zeros_like(per_class_total)
    ).tolist()

    # Compute the mean per-class accuracy
    mean_per_class_acc = sum(per_class_acc) / len(per_class_acc)

    # Print overall metrics and the mean per-class accuracy in the same line
    print('* Mean per-class accuracy: {mean_pca:.3f}  Acc@1 {top1.global_avg:.3f}  Acc@3 {top3.global_avg:.3f}  '
          'Loss {losses.global_avg:.3f}'
          .format(top1=metric_logger.acc1, top3=metric_logger.acc3,
                  losses=metric_logger.loss, mean_pca=mean_per_class_acc))

    overall_metrics = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    overall_metrics['pca'] = mean_per_class_acc
    return overall_metrics

