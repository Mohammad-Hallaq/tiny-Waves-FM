from torch.utils.data import Dataset
import os
from PIL import Image
import torch
from pathlib import Path


class SpectrogramImages(Dataset):
    def __init__(self, root='', transform=None):
        self.root = root
        self.transform = transform
        self.img_files = os.listdir(self.root)

    def __getitem__(self, index):
        img = Image.open(os.path.join(self.root, self.img_files[index]))
        if self.transform:
            return self.transform(img), torch.tensor([1, ])
        else:
            return img, torch.tensor([1, ])

    def __len__(self):
        return len(self.img_files)
