import numpy as np
from torch.utils.data import Dataset
import h5py
import os
from torchvision.transforms import Compose, Resize, Lambda, Normalize, InterpolationMode
import torch


class PositioningNR(Dataset):
    def __init__(self, datapath, img_size=(224, 224)):
        self.img_size = img_size
        self.file_paths = [os.path.join(datapath, filename)
                           for filename in os.listdir(datapath) if 'outdoor' in filename]

        self.offsets = []

        # Create a global index for all samples across all files
        offset = 0
        for file_idx, path in enumerate(self.file_paths):
            with h5py.File(path, 'r') as data_file:
                num_samples = len(data_file['features'])
                self.offsets.append(num_samples + offset)
                offset += num_samples

        # Precompute stats (based on a subset if necessary)
        # self.min_val, self.max_val, self.mu, self.std, self.coord_min, self.coord_max = self.compute_stats()
        self.min_val, self.max_val = -0.096, 1.136
        self.mu = torch.as_tensor([0.4638, 0.4631, 0.4703, 0.4620])
        self.std = torch.as_tensor([0.1154, 0.1176, 0.0979, 0.1281])
        self.coord_nominal_min = torch.as_tensor([0, 0, 0], dtype=torch.float32)
        self.coord_nominal_max = torch.as_tensor([80, 60, 40], dtype=torch.float32)
        # Define transformations
        self.transforms = Compose([
            Lambda(lambda x: torch.as_tensor(x, dtype=torch.float32)),
            Resize(self.img_size, antialias=True, interpolation=InterpolationMode.BICUBIC),
            Lambda(lambda x: (x - self.min_val) / (self.max_val - self.min_val)),
            Normalize(self.mu, self.std)
        ])

    def compute_stats(self):
        data_files = [h5py.File(file_path) for file_path in self.file_paths]
        features = np.vstack([np.array(data_file['features']) for data_file in data_files])
        labels = np.vstack([np.array(data_file['labels']['position']) for data_file in data_files])
        transforms = Compose([Lambda(lambda x: torch.as_tensor(x, dtype=torch.float32)),
                              Resize(self.img_size, antialias=True, interpolation=InterpolationMode.BICUBIC)])
        features = transforms(features)
        min_val, max_val = torch.min(features).item(), torch.max(features).item()
        features = (features - min_val) / (max_val - min_val)
        mu = torch.mean(features, dim=(0, 2, 3))
        std = torch.std(features, dim=(0, 2, 3))
        coord_min = torch.as_tensor(np.min(labels, axis=0, keepdims=True))
        coord_max = torch.as_tensor(np.max(labels, axis=0, keepdims=True))
        return min_val, max_val, mu, std, coord_min, coord_max

    def __getitem__(self, index):
        # Map global index to specific file and local index
        if index >= self.offsets[-1]:
            raise IndexError

        for i, offset in enumerate(self.offsets):
            if index < offset:
                file_idx = i
                break

        if file_idx != 0:
            relative_index = index - self.offsets[file_idx - 1]
        else:
            relative_index = index

        # Lazy load features and labels
        with h5py.File(self.file_paths[file_idx], 'r') as data_file:
            features = data_file['features'][relative_index]
            labels = torch.as_tensor(data_file['labels']['position'][relative_index], dtype=torch.float)

        # Normalize labels
        labels = 2 * (labels - self.coord_nominal_min) / (self.coord_nominal_max - self.coord_nominal_min) - 1

        # Apply transformations to features
        features = self.transforms(features)
        return features, labels

    def __len__(self):
        return self.offsets[-1]

