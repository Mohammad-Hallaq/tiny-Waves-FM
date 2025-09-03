import argparse
from pathlib import Path
import os

import torch
import torch.nn.functional as F
from dataset_classes.csi_sensing import CSISensingDataset
from torch.utils.data import DataLoader
import models_vit
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns

from tqdm import tqdm
import numpy as np


def main(args):
    device = args.device
    data_dir = Path('downstream_tasks_datasets/NTU-Fi_HAR/test')
    models = [("vit_small_patch16", "sensing_small_75.pth")]
   
    batch_size = args.batch_size
    num_workers = args.num_workers

    test_set = CSISensingDataset(data_dir)
    test_loader = DataLoader(test_set, batch_size=batch_size, num_workers=num_workers, shuffle=False)
    accuracies = np.zeros((len(models),))
    conf_matrices = np.zeros((len(models), 6, 6))

    model = torch.load(Path('tasks_models/sensing/best_model.pth'), weights_only=False)

    with torch.no_grad():
        for i, (model_key, model_name) in enumerate(tqdm(models, desc="Models")):
            model = model.to(device)
            print(model)

            all_targets = []
            all_preds = []
            for k, (images, targets) in enumerate(tqdm(test_loader, desc="Batches", leave=False)):
                images = images.to(device)
                targets = targets.to(device)
                pred = model(images)
                pred = torch.argmax(pred, dim=-1)
                all_targets.extend(targets.tolist())
                all_preds.extend(pred.tolist())
            all_targets = np.array(all_targets)
            all_preds = np.array(all_preds)
            conf_matrices[i] = confusion_matrix(all_targets, all_preds)
            accuracies[i] = np.sum(all_targets == all_preds) / len(all_targets)

    # Save the accuracy array
    # np.save(args.output_conf_mats, conf_matrices)
    # np.save(args.output_accuracy, accuracies)
    for i in range(1):
        row_sums = np.sum(conf_matrices[i], axis=1)
        row_sums[row_sums == 0] = 1
        conf_matrices[i] = conf_matrices[i] / row_sums.astype(float)

    class_labels = test_set.labels
    titles = ['ViT-S75']

    plt.rcParams['font.family'] = 'serif'

    # Define a custom gridspec for the subplots
    fig = plt.figure(figsize=(14, 12))  # Increase figure width
    gs = fig.add_gridspec(1, 1, width_ratios=[1])  # Adjust the width ratio of the last column

    # Create subplots based on the gridspec
    axs = [fig.add_subplot(gs[i, j]) for i in range(1) for j in range(1)]

    for i, ax in enumerate(axs):
        if (i + 1) % 1 == 0:
            sns.heatmap(conf_matrices[i], annot=True, fmt='.2f', cmap='Reds',
                        xticklabels=class_labels, yticklabels=class_labels, ax=ax,
                        annot_kws={'size': 10})  # Add colorbar here
        else:
            sns.heatmap(conf_matrices[i], annot=True, fmt='.2f', cmap='Reds',
                        xticklabels=class_labels, yticklabels=class_labels, ax=ax,
                        annot_kws={'size': 10}, cbar=False)  # No colorbar for other subplots
        ax.set_title(titles[i], fontsize=16)
        ax.tick_params(axis='both', labelsize=10)

    # Adjust axis labels for the second row, first column (index 3) and third row, second column (index 7)
    axs[0].set_ylabel('True label', fontsize=16)
    axs[0].set_xlabel('Predicted label', fontsize=16)

    # Adjust layout to avoid overlap
    plt.tight_layout()

    # Save the plot
    plt.savefig(Path('Figures/conf_mat_sensing.png'), dpi=400)

    # Show the plot
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate ViT models on CSI sensing dataset with varying mask ratios.")
    parser.add_argument('--ckpt_dir', type=str, default='checkpoints', help='Directory for model checkpoints')
    parser.add_argument('--data_dir', type=str, default='../datasets/NTU-Fi_HAR/test',
                        help='Path to CSI Sensing dataset directory')
    parser.add_argument('--mask_ratios', type=int, nargs='+', default=[20, 30, 40, 50, 60, 65, 70, 75, 80, 85],
                        help='List of mask ratios to test')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for DataLoader')
    parser.add_argument('--num_workers', type=int, default=0, help='Number of workers for DataLoader')
    parser.add_argument('--plot', action='store_true', help='Flag to enable plotting of images')
    parser.add_argument('--output_plot', type=str, default='accuracy_vs_mask_ratio_sensing.png',
                        help='Path to save the accuracy plot')
    parser.add_argument('--output_accuracy', type=str, default='accuracies_sensing.npy',
                        help='Path to save the accuracy array as a .npy file')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')

    args = parser.parse_args()
    main(args)
