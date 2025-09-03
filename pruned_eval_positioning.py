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

model = torch.load('/home/ict317-3/Mohammad/Tiny-WFMs/pruned_results/positioning/best_model.pth', weights_only=False)

for m in model.modules():
            if isinstance(m, timm.models.vision_transformer.Attention):
                m.forward = forward.__get__(m, timm.models.vision_transformer.Attention)

                
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
plt.savefig('/pruning_results/positioning/hist_positioning.png', dpi=300)
plt.show()

# distances_train.sort()
# distances_test.sort()

# cdf_train = np.linspace(0, 1, len(dataset_train))
# cdf_test = np.linspace(0, 1, len(dataset_test))
# idx_90_train = np.argmin(np.abs(cdf_train - 0.1))
# idx_90_test = np.argmin(np.abs(cdf_test - 0.1))
#
# plt.rcParams['font.family'] = 'serif'
# fig, axs = plt.subplots(1, 1)
# axs.plot(distances_train, cdf_train, label='train', linewidth=2, color='r')
# axs.plot(distances_test, cdf_test, label='test', linewidth=2, color='b')
# axs.axhline(y=0.1, linewidth=1, linestyle='--', label='90% likely', color='k')
# axs.axvline(x=distances_train[idx_90_train], linewidth=1, linestyle='--', color='r', alpha=0.8)
# axs.axvline(x=distances_test[idx_90_test], linewidth=1, linestyle='--', color='b', alpha=0.8)
# axs.set_xlabel('Positioning Error (m)')
# axs.set_ylabel('CDF')
# axs.legend(loc='lower right')
# plt.tight_layout()
# # plt.savefig('cdf_positioning.png', dpi=300)
# plt.show()
