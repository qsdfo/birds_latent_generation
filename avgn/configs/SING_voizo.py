from pathlib import Path
import numpy as np
from torch import nn

# Hparams
sr = 16000
num_mel_bins = 64
win_length_ms = None
hop_length_ms = None
n_fft = 512
chunk_len_ms = 1000
mel_lower_edge_hertz = 1000
mel_upper_edge_hertz = 7900

#  Model
n_z = 32

config = {
    # --- Dataset ---
    # voizo_all_wl-None_hl-None_nfft-1024_pad-128_melb-64_mell-500_melh-8000_pow-1.5
    'dataset': 'voizo_all_test',
    'dataset_preprocessing': f'sr-{sr}_'
                             f'wl-{win_length_ms}_'
                             f'hl-{hop_length_ms}_'
                             f'nfft-{n_fft}_'
                             f'chunkLen-{chunk_len_ms}_'
                             f'melb-{num_mel_bins}_'
                             f'mell-{mel_lower_edge_hertz}_'
                             f'melh-{mel_upper_edge_hertz}_'
                             'pow-1.5',

    # --- Model ---
    'model_type': 'VAE',
    'n_z': n_z,
    'encoder_kwargs': dict(
        type='Conv_stack',
        conv_stack=[
            nn.Conv2d(1, 64, 3, stride=(2, 2), padding=1),  #  padding = (kernel_size-1)/2
            nn.Conv2d(64, 64, 3, stride=(2, 2), padding=1),
            nn.Conv2d(64, 64, 3, stride=(2, 2), padding=1),
        ],
        conv2z=[
            nn.Linear(in_features=(h_dim // (2 ** 3)) * (w_dim // (2 ** 3)) * 64, out_features=512),
            nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.Linear(in_features=512, out_features=n_z * 2)
        ],
    ),
    'decoder_kwargs': dict(),
    'model_kwargs': dict(
        beta=1.0
    ),

    # ======== Training ========
    'lr': 1e-4,
    'batch_size': 64,
    'num_batches': 1024,
    'num_epochs': 500000,

    # ======== model ID ========
    'timestamp': None,
    'savename': Path(__file__).stem,
}
