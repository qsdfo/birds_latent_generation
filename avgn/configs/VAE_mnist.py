from pathlib import Path
from torch import nn

deconv_input_shape = (64, 7, 7)

config = {
    # --- Model ---
    'model_type': 'VAE',
    'dataset': 'mnist',
    'n_z': 32,
    'decoder_kwargs': dict(
        deconv_input_shape=deconv_input_shape,
        deconv_stack=[
            nn.ConvTranspose2d(deconv_input_shape[0], 64, (2, 2), stride=(2, 2)),  # (b, 64, 14, 14)
            nn.ConvTranspose2d(64, 32, (2, 2), stride=(2, 2)),  # (b, 32, 28, 28)
            nn.ConvTranspose2d(32, 1, 1, stride=(1, 1))]  # merge channels
    ),
    'encoder_kwargs': dict(),
    'model_kwargs': dict(
        beta=2.0
    ),

    # --- Dataloader ---
    'dataloader_generator_kwargs': dict(),

    # --- DataProcessor ---
    'data_processor_kwargs': dict(),

    # ======== Training ========
    'lr': 1e-4,
    'batch_size': 64,
    'num_batches': 512,
    'num_epochs': 50,

    # ======== model ID ========
    'timestamp': None,
    'savename': Path(__file__).stem,
}
