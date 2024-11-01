import os
import re

import torch
from torchvision.transforms import Lambda, Compose, Resize, InterpolationMode
from torch.utils.data import Dataset
from scipy.io import loadmat
from pathlib import Path


class CSISensingDataset(Dataset):
    def __init__(self, root_dir, img_size=(224, 224)):
        self.root_dir = root_dir
        self.file_list = os.listdir(Path(root_dir))
        self.labels = ['run', 'walk', 'fall', 'box', 'circle', 'clean']
        self.transform = Compose([Lambda(lambda x: torch.as_tensor(x, dtype=torch.float32)),
                                  Resize(img_size, antialias=True, interpolation=InterpolationMode.BICUBIC)])

    @staticmethod
    def _split_at_number(s):
        match = re.match(r"([a-zA-Z]+)(\d+)", s)
        if match:
            return match.groups()[0]
        else:
            raise ValueError("String is not in correct format.")

    def __getitem__(self, index):
        sample_name = self.file_list[index]
        csi = loadmat(os.path.join(self.root_dir, sample_name))['CSIamp']
        csi = csi.reshape(3, 114, -1)
        label_name = self._split_at_number(sample_name)
        label_index = self.labels.index(label_name)
        return self.transform(csi), torch.as_tensor(label_index, dtype=torch.long)

    def __len__(self):
        return len(self.file_list)

