from torch.utils.data import Dataset
import os
from PIL import Image
import torch
from pathlib import Path
from torchvision.transforms import PILToTensor


class AMCImages(Dataset):
    def __init__(self, root, transform=PILToTensor(), get_snr=False):
        self.root = root
        self.transform = transform
        self.img_files = os.listdir(self.root)
        self.labels = ['BPSK', 'QPSK', '8PSK', 'QAM16', 'QAM64', 'AM-DSB',
                       'AM-SSB', 'GFSK', 'CPFSK', 'PAM4', 'WBFM']
        self.label_mapping = {label: idx for idx, label in enumerate(self.labels)}
        self.label_mapping.update({
            '16QAM': self.label_mapping['QAM16'],
            '64QAM': self.label_mapping['QAM64'],
            '4PAM': self.label_mapping['PAM4']
        })
        self.get_snr = get_snr

    def __getitem__(self, index):
        img = Image.open(os.path.join(self.root, self.img_files[index]))
        label, snr, _ = self.img_files[index].split('_')
        label_idx = self.label_mapping.get(label)
        if self.get_snr:
            return self.transform(img), torch.as_tensor(label_idx, dtype=torch.long), torch.as_tensor(int(snr), dtype=torch.int32)
        return self.transform(img), torch.as_tensor(label_idx, dtype=torch.long)

    def __len__(self):
        return len(self.img_files)
