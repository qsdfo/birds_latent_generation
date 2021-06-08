from pathlib import Path
import numpy as np
from torch import nn

# Spectros
sr = 44100
num_mel_bins = 128
time_dim = 1024
win_length = 1024
hop_length = 32
n_fft = 1024
chunkls = 33760
mel_lower_edge_hertz = 500
mel_upper_edge_hertz = 20000

#  Model
n_z = 32
deconv_input_shape = (256, 2, 8)  # (num_channel, x_dim_latent, y_dim_latent)
h_dim = num_mel_bins
w_dim = time_dim

config = {
    # --- Dataset ---
    'dataset': 'du-ra-mo-ni-ro_segmented_dataAug',
    'dataset_preprocessing': f'sr-{sr}_' \
                             f'wl-{win_length}_' \
                             f'hl-{hop_length}_' \
                             f'nfft-{n_fft}_' \
                             f'chunkls-{chunkls}_' \
                             f'melb-{num_mel_bins}_' \
                             f'mell-{mel_lower_edge_hertz}_' \
                             f'melh-{mel_upper_edge_hertz}',
    # --- Model ---
    'model_type': 'VAE',
    'n_z': n_z,
    'encoder_kwargs': dict(
        conv_stack=[
            #  padding = (kernel_size-1)/2
            nn.Conv2d(1, 256, 3, stride=(2, 2), padding=1),
            nn.Conv2d(256, 256, 3, stride=(2, 2), padding=1),
            nn.Conv2d(256, 256, 3, stride=(2, 2), padding=1),
            nn.Conv2d(256, 256, 3, stride=(2, 2), padding=1),
        ],
        conv2z=[
            nn.Linear(in_features=(h_dim // (2 ** 4)) * \
                      (w_dim // (2 ** 4)) * 256, out_features=2048),
            nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.Linear(in_features=2048, out_features=512),
            nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.Linear(in_features=512, out_features=n_z * 2)
        ],
    ),
    # Formula for deconv shape WITH NO DILATION = out_dim = in_dim * stride + (k_size - stride)
    # (be careful with checker board effect, use a kernel size multiple of the stride)
    'decoder_kwargs': dict(
        deconv_input_shape=deconv_input_shape,
        z2deconv=[
            nn.Linear(in_features=n_z, out_features=512),
            nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.Linear(in_features=512, out_features=2048),
            nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.Linear(in_features=2048, out_features=int(
                np.prod(deconv_input_shape))),
            nn.ReLU(),
            nn.Dropout(p=0.1),
        ],
        deconv_stack=[  # (b, 256, 2, 8)
            nn.ConvTranspose2d(deconv_input_shape[0], 256, (4, 4), stride=(4, 4)),  # (b, 256, 4, 32)
            nn.ConvTranspose2d(256, 256, (4, 4), stride=(2, 2)),  # (, , 8, 64)
            nn.ConvTranspose2d(256, 256, (4, 4), stride=(2, 2)),  # (, , 16, 128)
            nn.ConvTranspose2d(256, 256, (4, 4), stride=(2, 2)),  # (, , 32, 256)
            nn.ConvTranspose2d(256, 256, (4, 4), stride=(2, 2)),  # (, , 64, 512)
            nn.ConvTranspose2d(256, 1, (1, 1), stride=(1, 1))
        ]
    ),
    'model_kwargs': dict(
        beta=1.0
    ),

    # ======== Training ========
    'lr': 1e-4,
    'batch_size': 16,
    'num_batches': 1024,
    'num_epochs': 500000,

    # ======== model ID ========
    'timestamp': None,
    'savename': Path(__file__).stem,
}
