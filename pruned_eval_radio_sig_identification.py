import argparse
import os
from tqdm import tqdm

from torch.utils.data import DataLoader

from dataset_classes.radio_sig import RadioSignal
import torch
import models_vit
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import confusion_matrix
import random
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from pruned_engine_finetune_regression import forward
import timm


plt.rcParams['font.family'] = 'serif'

def main(args):
    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    # dataset = RadioSignal(Path('../datasets/radio_sig_identification'))
    dataset_train = RadioSignal(Path('downstream_tasks_datasets/radio_sig_identification/train'))
    dataset_test = RadioSignal(Path('downstream_tasks_datasets/radio_sig_identification/test'))

    # splitter = StratifiedShuffleSplit(n_splits=1, train_size=0.8, test_size=0.2, random_state=seed)
    # all_labels = [dataset[i][1] for i in range(len(dataset))]
    #
    # for train_idx, test_idx in splitter.split(range(len(dataset)), all_labels):
    #     dataset_train = torch.utils.data.Subset(dataset, train_idx)
    #     dataset_test = torch.utils.data.Subset(dataset, test_idx)

    dataloader_train = DataLoader(dataset_train, batch_size=256, shuffle=True, num_workers=0)
    dataloader_test = DataLoader(dataset_test, batch_size=256, shuffle=False, num_workers=0)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # model = models_vit.__dict__['vit_small_patch16'](global_pool='token', num_classes=20, drop_path_rate=0.1, in_chans=1, head_layers=1)
    # checkpoint = torch.load(Path('output_dir/checkpoint-299.pth'), map_location='cpu', weights_only=False)
    # msg = model.load_state_dict(checkpoint['model'], strict=True)
    # print(msg)

    tail = args.model_path.partition(".")[0]
    p_ratio = ''.join(ch for ch in tail[-3:-1] if ch.isdigit()) or "xx"

    model = torch.load(Path(args.model_path), weights_only=False)

    for m in model.modules():
                if isinstance(m, timm.models.vision_transformer.Attention):
                    m.forward = forward.__get__(m, timm.models.vision_transformer.Attention)

    if len(model.blocks) != 12:
        conf_dir = os.path.join(args.save_dir, f'results_for_{len(model.blocks)}_blocks')
    else:
        conf_dir = os.path.join(args.save_dir)

    os.makedirs(conf_dir, exist_ok=True)
    
    model = model.to(device)
    model.eval()

    class_names = ['ads-b', 'airband', 'ais', 'apt', 'bluetooth', 'cellular', 'dab', 'dsd', 'fm', 'lora', 'morse', 'on-off-keying','packet', 'pocsag', 'Radioteletype', 'rke', 'RS41-Radiosonde', 'sstv', 'vor', 'wifi']
    all_labels_train = []
    all_preds_train = []

    with torch.no_grad():
        for samples, targets in tqdm(dataloader_train, desc='Train batch'):
            samples, targets = samples.to(device), targets.to(device)
            output = model(samples)
            all_preds_train.extend(output.argmax(dim=-1).cpu().numpy())
            all_labels_train.extend(targets.cpu().numpy())

    all_labels_test = []
    all_preds_test = []

    with torch.no_grad():
        for samples, targets in tqdm(dataloader_test, desc='Test batch'):
            samples, targets = samples.to(device), targets.to(device)
            output = model(samples)
            all_preds_test.extend(output.argmax(dim=-1).cpu().numpy())
            all_labels_test.extend(targets.cpu().numpy())

    conf_mat_train = confusion_matrix(all_labels_train, all_preds_train)
    conf_mat_test = confusion_matrix(all_labels_test, all_preds_test)


    conf_mat_train = conf_mat_train.astype(np.float32)
    conf_mat_test = conf_mat_test.astype(np.float32)
    for i in range(len(class_names)):
        conf_mat_train[i] /= np.sum(conf_mat_train[i])
        conf_mat_test[i] /= np.sum(conf_mat_test[i])

    accuracy_train = np.mean(np.diagonal(conf_mat_train))
    accuracy_test = np.mean(np.diagonal(conf_mat_test))

    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    fig.suptitle('Pretrained with Positioning Data')
    # fig.suptitle('Supervised Learning Baseline')
    sns.heatmap(conf_mat_train, cmap='Reds', yticklabels=class_names, ax=axs[0])
    sns.heatmap(conf_mat_test, cmap='Reds',  yticklabels=class_names, ax=axs[1])
    axs[0].tick_params(axis='y', labelsize=10)
    axs[1].tick_params(axis='y', labelsize=10)
    axs[0].tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
    axs[1].tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
    axs[0].set_title(f'Train - Accuracy: {accuracy_train:.2f}')
    axs[1].set_title(f'Test - Accuracy: {accuracy_test:.2f}')
    plt.tight_layout()
    plt.savefig(os.path.join(conf_dir, f'conf_mat_radio_identification_{p_ratio}%.png'), dpi=400)
    plt.show()

if __name__ == '__main__':

    parser = argparse.ArgumentParser('Evaluation of Radio Signal Identification', add_help=True)
    parser.add_argument('--model_path', default='', type=str, metavar='MODEL',
                        help='Path to the pruned model')
    parser.add_argument('--save_dir', type=str, default='pruning_results/sig_identification',
                        help='Directory to save the histogram.')
    args = parser.parse_args()
    main(args)