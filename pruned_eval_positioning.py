
import argparse
import os
from tqdm import tqdm
import models_vit
from dataset_classes.positioning import Positioning5G
import torch
import numpy as np
import matplotlib.pyplot as plt
import random
from pathlib import Path
from torch.utils.data import DataLoader
from pruned_engine_finetune_regression import forward
import timm

def reverse_normalize(x, coord_min, coord_max):
    return (x + 1) / 2 * (coord_max - coord_min) + coord_min

   
def main(args):
    # Set seeds for reproducibility
    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    scene = 'outdoor'
    dataset_train = Positioning5G(Path('downstream_tasks_datasets/5G_NR_Positioning/outdoor/train'), scene=scene)
    dataset_test = Positioning5G(Path('downstream_tasks_datasets/5G_NR_Positioning/outdoor/test'), scene=scene)

    # Get coordinate tensors and reshape them appropriately
    coord_min = dataset_train.coord_nominal_min.view((1, -1))
    coord_max = dataset_train.coord_nominal_max.view((1, -1))

    dataloader_train = DataLoader(dataset_train, batch_size=128, shuffle=False, num_workers=0)
    dataloader_test = DataLoader(dataset_test, batch_size=128, shuffle=False, num_workers=0)

    # Choose device (GPU if available)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # model = models_vit.__dict__['vit_small_patch16'](
    #     global_pool='token', num_classes=3, drop_path_rate=0.1, in_chans=4)
    # checkpoint = torch.load(Path('output_dir_outdoor_small/checkpoint-0.pth'), map_location='cpu')
    # msg = model.load_state_dict(checkpoint['model'], strict=True)
    # print(msg)

    tail = args.model_path.partition(".")[0]
    p_ratio = ''.join(ch for ch in tail[-3:-1] if ch.isdigit()) or "xx"

    model = torch.load(Path(args.model_path), weights_only=False)

    for m in model.modules():
                if isinstance(m, timm.models.vision_transformer.Attention):
                    m.forward = forward.__get__(m, timm.models.vision_transformer.Attention)


    if len(model.blocks) != 12:
        hist_dir = os.path.join(args.save_dir, f'results_for_{len(model.blocks)}_blocks')
    else:
        hist_dir = os.path.join(args.save_dir)

    os.makedirs(hist_dir, exist_ok=True)
    
    # Move model to the chosen device
    model = model.to(device)

    # Move coordinate tensors to the chosen device
    coord_min = coord_min.to(device)
    coord_max = coord_max.to(device)

    # Create tensors to hold the distance errors on the chosen device
    distances_train = torch.zeros(len(dataset_train), device=device)
    with torch.no_grad():
        for i, batch in tqdm(enumerate(dataloader_train), desc='Train Batch', total=len(dataloader_train)):
            image, target = batch
            image = image.to(device)
            target = target.to(device)
            # Compute predictions and reverse normalize on the same device
            pred_position = reverse_normalize(model(image), coord_min, coord_max)
            true_position = reverse_normalize(target, coord_min, coord_max)
            num_samples = target.shape[0]
            # Compute Euclidean distance error for each sample (square root after sum)
            error = torch.sqrt(torch.sum((pred_position - true_position) ** 2, dim=1))
            distances_train[i * num_samples: (i + 1) * num_samples] = error

    distances_test = torch.zeros(len(dataset_test), device=device)
    with torch.no_grad():
        for i, batch in tqdm(enumerate(dataloader_test), desc='Test Batch', total=len(dataloader_test)):
            image, target = batch
            image = image.to(device)
            target = target.to(device)
            pred_position = reverse_normalize(model(image), coord_min, coord_max)
            true_position = reverse_normalize(target, coord_min, coord_max)
            num_samples = target.shape[0]
            error = torch.sqrt(torch.sum((pred_position - true_position) ** 2, dim=1))
            distances_test[i * num_samples: (i + 1) * num_samples] = error

    # Convert distance tensors to NumPy arrays for plotting
    distances_train_np = distances_train.cpu().numpy()
    distances_test_np = distances_test.cpu().numpy()

    plt.rcParams['font.family'] = 'serif'
    mean_train = np.mean(distances_train_np)
    mean_test = np.mean(distances_test_np)
    stdev_train = np.std(distances_train_np)
    stdev_test = np.std(distances_test_np)

    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    fig.suptitle('Pretrained with Spectrograms', fontsize=20)
    bins = 25

    axs[0].hist(distances_train_np, bins=bins, color='red', edgecolor='w', alpha=0.7, density=True)
    axs[0].axvline(mean_train, color='black', linestyle='--', linewidth=2,
                label=f'Mean: {mean_train:.2f} (m)\nStdev: {stdev_train:.2f} (m)')
    axs[0].set_xlabel('Positioning Error (m)', fontsize=16)
    axs[0].set_ylabel('Probability Density', fontsize=16)
    axs[0].legend(fontsize=16)

    axs[1].hist(distances_test_np, bins=bins, color='blue', edgecolor='w', alpha=0.7, density=True)
    axs[1].axvline(mean_test, color='black', linestyle='--', linewidth=2,
                label=f'Mean: {mean_test:.2f} (m)\nStdev: {stdev_test:.2f} (m)')
    axs[1].set_xlabel('Positioning Error (m)', fontsize=16)
    axs[1].set_ylabel('Probability Density', fontsize=16)
    axs[1].legend(fontsize=16)

    plt.tight_layout()
    plt.savefig(os.path.join(hist_dir, f'hist_positioning_{p_ratio}%.png'), dpi=300)
    plt.show()

if __name__ == '__main__':

    parser = argparse.ArgumentParser('Evaluation of 5G NR Positioning', add_help=True)
    parser.add_argument('--model_path', default='', type=str, metavar='MODEL',
                        help='Path to the pruned model')
    parser.add_argument('--save_dir', type=str, default='pruning_results/positioning',
                        help='Directory to save the histogram.')
    args = parser.parse_args()
    main(args)
