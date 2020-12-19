import pickle

import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset


class SpectroDataset(Dataset):
    def __init__(self, syllable_paths):
        super(SpectroDataset, self).__init__()
        self.syllable_paths = syllable_paths

    def __len__(self):
        return len(self.syllable_paths)

    @staticmethod
    def process_mSp(mSp):
        x_np = np.array(mSp).astype(np.float32) / 255.
        return np.expand_dims(x_np, axis=0)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        fname = self.syllable_paths[idx]
        with open(fname, 'rb') as ff:
            data = pickle.load(ff)
        mSp = data['mSp']
        # conv in pytorch are
        # (batch, channel, height, width)
        sample = SpectroDataset.process_mSp(mSp)
        return {
            'input': sample,
            'target': sample,
            'label': data['label']
        }
