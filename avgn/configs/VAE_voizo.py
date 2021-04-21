from pathlib import Path
import numpy as np
from torch import nn

# Data processing
sr = 44100
num_mel_bins = 64
# num_mel_bins = 32
time_dim = 256
win_length = 512
hop_length = 128
n_fft = 512
mel_lower_edge_hertz = 500
mel_upper_edge_hertz = 16000
# chunk_len = {
#     'type': 'stft_win',
#     'value': time_dim
# }
chunk_len_win = 256
# Choose dataset
chunkls_max = 44100
assert ((chunk_len_win - 1) * hop_length + win_length) <= chunkls_max
suffix = f'sr-{sr}_chunklmaxs-{chunkls_max}_locut-{mel_lower_edge_hertz}_hicut-{mel_upper_edge_hertz}'

#  Model
n_z = 16
deconv_input_shape = (64, 7, 15)  # (num_channel, x_dim_latent, y_dim_latent)
# deconv_input_shape = (64, 3, 15)  # (num_channel, x_dim_latent, y_dim_latent)
h_dim = num_mel_bins
w_dim = time_dim

config = {
    # --- Dataset ---
    'dataset': f'voizo_all_segmented_{suffix}',
    'data_processing': {
        'sr': sr,
        'num_mel_bins': num_mel_bins,
        'n_fft': n_fft,
        'mel_lower_edge_hertz': mel_lower_edge_hertz,
        'mel_upper_edge_hertz': mel_upper_edge_hertz,
        'hop_length': hop_length,
        'win_length': win_length,
        'power': 1.5,
        'ref_level_db': -35,
        'preemphasis': 0.97,
        'chunk_len_win': chunk_len_win,
        #     mask_spec=False,
        #     mask_spec_kwargs={"spec_thresh": 0.9, "offset": 1e-10},
        #     reduce_noise=True,
    },
    # --- Model ---
    'model_type': 'VAE',
    'n_z': n_z,
    'encoder_kwargs': dict(
        conv_stack=[
            #  padding = (kernel_size-1)/2
            nn.Conv2d(1, 64, 3, stride=(2, 2), padding=1),
            nn.Conv2d(64, 64, 3, stride=(2, 2), padding=1),
            nn.Conv2d(64, 64, 3, stride=(2, 2), padding=1),
            nn.Conv2d(64, 64, 3, stride=(2, 2), padding=1),
        ],
        conv2z=[
            nn.Linear(in_features=(h_dim // (2 ** 4)) * \
                      (w_dim // (2 ** 4)) * 64, out_features=512),
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
            nn.Linear(in_features=512, out_features=int(
                np.prod(deconv_input_shape))),
            nn.ReLU(),
            nn.Dropout(p=0.1),
        ],
        # Deconv stack for 64 mel bands
        deconv_stack=[  # (b, 64, 7, 15)
            # (b, 64, 14, 30)
            nn.ConvTranspose2d(
                deconv_input_shape[0], 64, (2, 2), stride=(2, 2)),
            nn.ConvTranspose2d(64, 64, (4, 4), stride=(
                2, 2), output_padding=1),  # (, , 31, 63)
            nn.ConvTranspose2d(64, 64, (4, 4), stride=(2, 2)),  # (, , 64, 128)
            nn.ConvTranspose2d(64, 64, (1, 2), stride=(1, 2)),  # (, , 64, 256)
            nn.ConvTranspose2d(64, 1, (1, 1), stride=(1, 1))
        ]
        # Deconv stack for 32 mel bands
        # deconv_stack=[  # (b, 64, 3, 15)
        #     nn.ConvTranspose2d(deconv_input_shape[0], 64, (2, 2), stride=(2, 2)),  # (b, 64, 6, 30)
        #     nn.ConvTranspose2d(64, 64, (4, 4), stride=(2, 2), output_padding=1),  # (, , 15, 63)
        #     nn.ConvTranspose2d(64, 64, (4, 4), stride=(2, 2)),  # (, , 32, 128)
        #     nn.ConvTranspose2d(64, 64, (1, 2), stride=(1, 2)),  # (, , 32, 256)
        #     nn.ConvTranspose2d(64, 1, (1, 1), stride=(1, 1))
        # ]
    ),
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
