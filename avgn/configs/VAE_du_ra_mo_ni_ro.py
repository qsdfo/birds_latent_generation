from pathlib import Path
import numpy as np
from torch import nn

# Spectros
sr = 44100
num_mel_bins = 64
time_dim = 256
win_length = 512
hop_length = 128
n_fft = 512
chunkls = 33152
mel_lower_edge_hertz = 500
mel_upper_edge_hertz = 20000

#  Model
n_z = 16
deconv_input_shape = (256, 2, 5)  # (num_channel, x_dim_latent, y_dim_latent)
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
                      (w_dim // (2 ** 4)) * 256, out_features=1024),
            nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.Linear(in_features=1024, out_features=n_z * 2)
        ],
    ),
    # Formula for deconv shape WITH NO DILATION = out_dim = in_dim * stride + (k_size - stride)
    # (be careful with checker board effect, use a kernel size multiple of the stride)
    'decoder_kwargs': dict(
        deconv_input_shape=deconv_input_shape,
        z2deconv=[
            nn.Linear(in_features=n_z, out_features=1024),
            nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.Linear(in_features=1024, out_features=int(
                np.prod(deconv_input_shape))),
            nn.ReLU(),
            nn.Dropout(p=0.1),
        ],
        deconv_stack=[  # (b, 256, 1, 8)
            # nn.ConvTranspose2d(deconv_input_shape[0], 256, (1, 2), stride=(1, 2)),  # (b, 256, 2, 14)
            # nn.ConvTranspose2d(256, 256, (3, 3), stride=(3, 3)),  # (b, 256, 6, 42)
            # nn.ConvTranspose2d(256, 256, (4, 3), stride=(2, 3)),  # (b, 256, 14, 126)
            # nn.ConvTranspose2d(256, 256, (4, 4), stride=(2, 2)),  # (, , 30, 254)
            # nn.ConvTranspose2d(256, 256, (4, 4), stride=(2, 2), output_padding=1),  # (, , 63, 511)
            # nn.ConvTranspose2d(256, 256, (4, 4), stride=(2, 2)),  # (, , 128, 1024)
            # nn.ConvTranspose2d(256, 1, (1, 1), stride=(1, 1))
            #
            # nn.ConvTranspose2d(deconv_input_shape[0], 256, (2, 2), stride=(2, 2)),  # (b, 64, 14, 30)
            # nn.ConvTranspose2d(256, 256, (4, 4), stride=(2, 2), output_padding=1),  # (, , 31, 63)
            # nn.ConvTranspose2d(256, 256, (4, 4), stride=(2, 2)),  # (, , 64, 128)
            # nn.ConvTranspose2d(256, 256, (1, 2), stride=(1, 2)),  # (, , 64, 256)
            # nn.ConvTranspose2d(256, 1, (1, 1), stride=(1, 1))
            #
            # # (, , 2, 5) -> (, , 4, 20)
            nn.ConvTranspose2d(
                deconv_input_shape[0], 256, (2, 4), stride=(2, 4)),
            nn.ConvTranspose2d(256, 256, (2, 2), stride=(1, 1)),  # (, , 5, 21)
            nn.ConvTranspose2d(
                256, 256, (3, 3), stride=(3, 3)),  # (, , 15, 63)
            nn.ConvTranspose2d(
                256, 256, (2, 2), stride=(1, 1)),  # (, , 16, 64)
            nn.ConvTranspose2d(
                256, 256, (2, 2), stride=(2, 2)),  # (, , 32, 128)
            nn.ConvTranspose2d(
                256, 256, (2, 2), stride=(2, 2)),  # (, , 64, 256)
            nn.ConvTranspose2d(256, 1, (1, 1), stride=(1, 1))
        ]
    ),
    'model_kwargs': dict(
        beta=1.0
    ),

    # ======== Training ========
    'lr': 1e-4,
    'batch_size': 8,
    'num_batches': 512,
    'num_epochs': 500000,

    # ======== model ID ========
    'timestamp': None,
    'savename': Path(__file__).stem,
}
