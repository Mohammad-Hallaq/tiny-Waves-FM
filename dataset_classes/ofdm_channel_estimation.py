import os
import numpy as np
import torch

from torch.utils.data import Dataset
from torchvision.transforms import Compose, Lambda, Normalize, Resize, InterpolationMode


class OfdmChannelEstimation(Dataset):
    def __init__(self, data_path, normalize_labels=False):
        self.data_path = data_path
        self.file_list = [os.path.join(self.data_path, file_name) for file_name in os.listdir(self.data_path)]

        # statistics
        self.min_features, self.max_features = -12.33, 13.09
        self.min_label, self.max_label = -6.49, 6.34
        self.mu = [-0.0297, -0.0304]
        self.std = [0.0953, 0.0953]

        self.transform = Compose([Lambda(lambda x: torch.as_tensor(x, dtype=torch.float32)),
                                  Lambda(lambda x: 2 * (x - self.min_features) / (self.max_features - self.min_features) - 1),
                                  Normalize(mean=self.mu, std=self.std),
                                  Resize((224, 224), interpolation=InterpolationMode.BICUBIC, antialias=True),])
        if normalize_labels:
            self.transform_label = Compose([Lambda(lambda x: torch.as_tensor(x, dtype=torch.float32)),
                                            Lambda(lambda x: 2 * (x - self.min_label) / (self.max_label - self.min_label) - 1)])
        else:
            self.transform_label = Lambda(lambda x: torch.as_tensor(x, dtype=torch.float32))

    def create_sample(self, x_rg, y_rg):
        x_rg_pilot = np.concatenate((x_rg[:, 2], x_rg[:, 11]), axis=1)
        x_rg_pilot = np.stack((x_rg_pilot.real, x_rg_pilot.imag), axis=1).reshape((x_rg_pilot.shape[0], 2, 1, -1))
        y_rg_pilot = np.concatenate((y_rg[:, :, 2], y_rg[:, :, 11]), axis=2)
        y_rg_pilot = np.stack((y_rg_pilot.real, y_rg_pilot.imag), axis=1)
        x = np.concatenate((x_rg_pilot, y_rg_pilot), axis=2)
        return self.transform(x)

    def __getitem__(self, index):
        with np.load(self.file_list[index]) as data:
            x = data['x']
            h = data['h']
            snr_db = float(data['snr_db'])
        return self.transform(x), self.transform_label(h), torch.as_tensor(snr_db, dtype=torch.float32)

    def reverse_normalize(self, h):
        h = (h + 1) / 2 * (self.max_label - self.min_label) + self.min_label
        return h

    def __len__(self):
        return len(self.file_list)
