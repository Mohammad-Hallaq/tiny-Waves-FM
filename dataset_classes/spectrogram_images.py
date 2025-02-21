from torch.utils.data import Dataset
import os
from PIL import Image
import torch


class SpectrogramImages(Dataset):
    def __init__(self, roots, transform=None):
        """
        Args:
            roots (str or list of str): Directory path or list of directory paths containing images.
            transform (callable, optional): A function/transform to apply to each image.
        """
        # Ensure roots is a list even if a single directory is provided
        if isinstance(roots, str):
            roots = [roots]
        self.roots = roots
        self.transform = transform
        self.img_files = []
        for root in self.roots:
            # Get full file path for each image in the directory
            for file in os.listdir(root):
                self.img_files.append(os.path.join(root, file))

    def __getitem__(self, index):
        # Open the image from the full path
        img = Image.open(self.img_files[index])
        if self.transform:
            return self.transform(img), torch.tensor([1])
        else:
            return img, torch.tensor([1])

    def __len__(self):
        return len(self.img_files)
