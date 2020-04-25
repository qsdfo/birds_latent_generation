import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset


class SpectroDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, syllable_df):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        super(SpectroDataset, self).__init__()
        self.syllable_df = syllable_df

    def __len__(self):
        return len(self.syllable_df)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        x_np = np.array(self.syllable_df.spectrogram[idx]).astype(np.float32) / 255.
        # conv in pytorch are
        # (batch, channel, height, width)
        sample = np.expand_dims(x_np, axis=0)
        return sample