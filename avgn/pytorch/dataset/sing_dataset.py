import pickle

import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset


class SingDataset(Dataset):
    def __init__(self, syllable_paths):
        super(SingDataset, self).__init__()
        self.win_len = 1024
        self.hop_len = 256
        self.prepare_items(syllable_paths)

    def prepare_items(self, syllable_paths):
        """
        Create indexing for chunks of the training size
        """
        self.idx_to_chunk = []
        for syllable_path in syllable_paths:
            with open(syllable_path, 'rb') as ff:
                data = pickle.load(ff)
            sn = data['sn']
            for start_t in range(0, len(sn) - self.hop_len, self.hop_len):
                chunk = sn[start_t:start_t + self.win_len + self.hop_len]
                # Remove chunks containing only zeros
                if np.any(chunk != 0):
                    self.idx_to_chunk.append({
                        'path': syllable_path,
                        'start_t': start_t
                    })

    @staticmethod
    def process_mSp(mSp):
        mSp_np = np.array(mSp)
        if mSp_np.max() <= 1:
            x_np = mSp_np.astype(np.float32)
        else:
            x_np = mSp_np.astype(np.float32) / 255.
        return np.expand_dims(x_np, axis=0)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        chunk_info = self.idx_to_chunk[idx]
        with open(chunk_info['path'], 'rb') as ff:
            data = pickle.load(ff)
        sn = data['sn']
        # conv in pytorch are
        # (batch, channel, height, width)
        mS = process_mSp(mS_int)
        return {
            'input': mS,
            'target': sn,
            'label': data['label']
        }
