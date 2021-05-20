import pickle
import numpy as np
import torch
from torch.utils.data import Dataset


class SpectroDataset(Dataset):
    def __init__(self, syllable_paths, chunk_len_win, num_mel_bins):
        super(SpectroDataset, self).__init__()
        self.syllable_paths = syllable_paths
        self.chunk_len_win = chunk_len_win
        self.num_mel_bins = num_mel_bins

    def __len__(self):
        return len(self.syllable_paths)

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
        fname = self.syllable_paths[idx]
        with open(fname, 'rb') as ff:
            data = pickle.load(ff)
        mSp = data['mS_int']

        # Pad
        win_len = mSp.shape[1]
        if win_len < self.chunk_len_win:
            pad_size = (self.chunk_len_win - win_len) // 2
            mSp_pad = np.zeros(
                (self.num_mel_bins, self.chunk_len_win))
            mSp_pad[:, pad_size:(pad_size + win_len)] = mSp
        else:
            mSp_pad = mSp[:, :self.data_processing['chunk_len_win']]

        # conv in pytorch are
        # (batch, channel, height, width)
        sample = SpectroDataset.process_mSp(mSp_pad)
        return {
            'input': sample,
            'target': sample,
            'label': data['label']
        }
