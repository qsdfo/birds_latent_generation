# Birds latent generation

Bird song synthesizer based on VAE and ConvNets.
Heavily relies on data augmentations for now.

## Process
- main_download
- chunk files? (in scripting)
- you may want to run main_calibrate_db to find the ideal values for the ref and min dB parameters
- main_preprocess (create json files)
- main_segment (useless if manually annotated beforehand)
- main_spectrogram
- train and generate with main_torch
