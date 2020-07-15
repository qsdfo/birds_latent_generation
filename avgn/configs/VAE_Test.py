from pathlib import Path
from torch import nn

h_dim = 128
w_dim = 128
deconv_input_shape = (64, 8, 8)

win_length_ms = None
hop_length_ms = 10
n_fft = 1024
num_mel_bins = 64
mel_lower_edge_hertz = 500
mel_upper_edge_hertz = 12000

config = {
    # --- Dataset ---
    'dataset': 'Test',
    'dataset_preprocessing': f'wl{win_length_ms}_'
                             f'hl{hop_length_ms}_'
                             f'nfft{n_fft}_'
                             f'melb{num_mel_bins}_'
                             f'mell{mel_lower_edge_hertz}_'
                             f'melh{mel_upper_edge_hertz}',

    # --- Model ---
    'model_type': 'VAE',
    'n_z': 32,
    'encoder_kwargs': dict(
        conv_stack=[
            nn.Conv2d(1, 32, 2, stride=(2, 2)),
            nn.Conv2d(32, 64, 2, stride=(2, 2)),
            nn.Conv2d(64, 64, 2, stride=(2, 2)),
            nn.Conv2d(64, 64, 2, stride=(2, 2)),
        ],
        out_stack_dim=(h_dim // (2 ** 4)) * (w_dim // (2 ** 4)) * 64
    ),
    'decoder_kwargs': dict(
        deconv_input_shape=deconv_input_shape,
        deconv_stack=[nn.ConvTranspose2d(deconv_input_shape[0], 64, (2, 2), stride=(2, 2)),  # (b, 64, 16,16)
                      nn.ConvTranspose2d(deconv_input_shape[0], 64, (2, 2), stride=(2, 2)),  # (b, 64, 32, 32)
                      nn.ConvTranspose2d(64, 64, (2, 2), stride=(2, 2)),  # (b, 32, 64, 64)
                      nn.ConvTranspose2d(64, 32, (2, 2), stride=(2, 2)),  # (b, 32, 128, 128)
                      nn.ConvTranspose2d(32, 1, 1, stride=(1, 1))]
    ),
    'model_kwargs': dict(
        beta=1.0
    ),

    # ======== Training ========
    'lr': 1e-4,
    'batch_size': 64,
    'num_batches': 512,
    'num_epochs': 500000,

    # ======== model ID ========
    'timestamp': None,
    'savename': Path(__file__).stem,
}
