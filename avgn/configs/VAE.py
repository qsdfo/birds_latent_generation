from pathlib import Path


config = {
    # --- Model ---
    'model_type': 'VAE',
    'n_z': 32,
    'decoder_kwargs': dict(),
    'encoder_kwargs': dict(),
    'model_kwargs': dict(),

    # --- Dataloader ---
    'dataloader_generator_kwargs': dict(),

    # --- DataProcessor ---
    'data_processor_kwargs':       dict(),

    # ======== Training ========
    'lr':                          1e-4,
    'batch_size':                  3,
    'num_epochs':                  500,

    # ======== model ID ========
    'timestamp':                   None,
    'savename':                    Path(__file__).stem,
}
