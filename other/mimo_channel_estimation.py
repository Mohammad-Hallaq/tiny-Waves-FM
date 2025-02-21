import pickle
from scipy.signal import ShortTimeFFT, windows

import torch
from torch.utils.data import Dataset
from pathlib import Path
from torchvision.transforms import ToTensor, Compose, Resize, InterpolationMode, Lambda
from torchvision.transforms.functional import resize
import numpy as np


n_subcarriers = 320
mfft = 64
hop = mfft // 2
win = windows.hann(mfft)
SFT = ShortTimeFFT(win=win, hop=hop, fs=1, scale_to='magnitude', fft_mode='centered', mfft=mfft)


class MIMOChannelEstimation(Dataset):
    def __init__(self, data_path):
        self.data_path = data_path
        # Load data once during initialization
        with open(self.data_path, 'rb') as f:
            self.data = pickle.load(f)
        self.preamble = resize(torch.from_numpy(self.data['P']).float().view((1, 32, 32)), [224, 224], antialias=True,
                               interpolation=InterpolationMode.BICUBIC)
        self.preamble = (self.preamble - torch.mean(self.preamble)) / torch.std(self.preamble)
        self.num_samples = self.data['y']['real'].shape[0]
        self.transform = Compose([ToTensor(),
                                  Resize((224, 224), InterpolationMode.BICUBIC),
                                  Lambda(lambda x: (x - x.mean()) / x.std())])

    def __getitem__(self, index):
        # Access preloaded data
        real_part = torch.from_numpy(self.data['y']['real'][index]).float()
        imag_part = torch.from_numpy(self.data['y']['imag'][index]).float()
        y = torch.stack((real_part, imag_part), dim=0)

        preamble_key, i = self.data['X'][index]
        rx_preamble = (self.data['LTF'][preamble_key]['real'][i * n_subcarriers: (i + 1) * n_subcarriers]
                       + 1j * self.data['LTF'][preamble_key]['imag'][i * n_subcarriers: (i + 1) * n_subcarriers])
        rx_preamble_spectrogram = 10 * np.log10(SFT.spectrogram(rx_preamble))
        x = torch.cat((self.preamble, self.transform(rx_preamble_spectrogram)), dim=0).float()
        return x, y

    def __len__(self):
        return self.num_samples

