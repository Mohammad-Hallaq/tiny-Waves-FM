import argparse
from pathlib import Path
import os

import torch
from dataset_classes.csi_sensing import CSISensingDataset
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader
import models_vit
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import numpy as np

from pruned_engine_finetune_regression import forward
import timm

def main(args):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data_dir = Path('downstream_tasks_datasets/NTU-Fi_HAR/test')
    models = [("vit_small_patch16", "sensing_small_75.pth")]

    test_set = CSISensingDataset(data_dir)
    test_loader = DataLoader(test_set, batch_size=256, num_workers=0, shuffle=False)
    accuracies = np.zeros((len(models),))
    conf_matrices = np.zeros((len(models), 6, 6))

    tail = args.model_path.partition(".")[0]
    p_ratio = ''.join(ch for ch in tail[-3:-1] if ch.isdigit()) or "xx"

    model = torch.load(args.model_path, weights_only=False)
    
    for m in model.modules():
            if isinstance(m, timm.models.vision_transformer.Attention):
                m.forward = forward.__get__(m, timm.models.vision_transformer.Attention)


    if len(model.blocks) != 12:
        conf_dir = os.path.join(args.save_dir, f'results_for_{len(model.blocks)}_blocks')
    else:
        conf_dir = os.path.join(args.save_dir)

    os.makedirs(conf_dir, exist_ok=True)

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
    plt.savefig(os.path.join(conf_dir, f'conf_mat_sensing_{p_ratio}%.png'), dpi=400)

    # Show the plot
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate ViT models on CSI sensing dataset with varying mask ratios.")
    parser.add_argument('--model_path', default='', type=str, metavar='MODEL',
                        help='Path to the pruned model')
    parser.add_argument('--save_dir', type=str, default='pruning_results/sensing',
                        help='Directory to save the histogram.')
    args = parser.parse_args()
    main(args)