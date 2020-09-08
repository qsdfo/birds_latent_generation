from pathlib import Path
import numpy as np
from torch import nn

# Spectros
h_dim = 64
w_dim = 64
win_length_ms = None
hop_length_ms = None
n_fft = 1024
num_mel_bins = 64
mel_lower_edge_hertz = 500
mel_upper_edge_hertz = 8000

#  Model
n_z = 32
deconv_input_shape = (64, 6, 6)
config = {
    # --- Dataset ---
    'dataset': 'Bird_all',
    'dataset_preprocessing': f'wl-{win_length_ms}_'
                             f'hl-{hop_length_ms}_'
                             f'nfft-{n_fft}_'
                             f'melb-{num_mel_bins}_'
                             f'mell-{mel_lower_edge_hertz}_'
                             f'melh-{mel_upper_edge_hertz}_'
                             f'pow-1.5',

    # --- Model ---
    'model_type': 'VAE',
    'n_z': n_z,
    'encoder_kwargs': dict(
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
    'decoder_kwargs': dict(
        deconv_input_shape=deconv_input_shape,
        z2deconv=[
            nn.Linear(in_features=n_z, out_features=512),
            nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.Linear(in_features=512, out_features=int(np.prod(deconv_input_shape))),
            nn.ReLU(),
            nn.Dropout(p=0.1),
        ],
        deconv_stack=[
            nn.ConvTranspose2d(deconv_input_shape[0], 64, (4, 4), stride=(2, 2), output_padding=1),  # (b, 64, 14, 14)
            nn.ConvTranspose2d(64, 64, (4, 4), stride=(2, 2)),  # (, , 32, 32)
            nn.ConvTranspose2d(64, 64, (2, 2), stride=(2, 2)),  # (, , 64, 64)
            nn.ConvTranspose2d(64, 1, 1, stride=(1, 1))
        ]
    ),
    'model_kwargs': dict(
        beta=1.0
    ),

    # ======== Training ========
    'lr': 1e-4,
    'batch_size': 512,
    'num_batches': 1024,
    'num_epochs': 500000,

    # ======== model ID ========
    'timestamp': None,
    'savename': Path(__file__).stem,
}

