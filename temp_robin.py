# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %% [markdown]
# # Contamination

# %%
from avgn.pytorch.dataset.spectro_dataset import SpectroDataset
from avgn.pytorch.generate.interpolation import constant_radius_interpolation
import random
from main_spectrogramming import process_syllable
from avgn.signalprocessing.dynamic_thresholding_scipy import dynamic_threshold_segmentation
from avgn.signalprocessing.create_spectrogram_dataset import prepare_wav
import os
from avgn.pytorch.getters import get_model_and_dataset
from avgn.signalprocessing.spectrogramming_scipy import _db_to_amplitude, _denormalize, _mel_to_linear, _min_level_db, build_mel_basis, build_mel_inversion_basis, inv_spectrogram_sp, griffinlim_sp
from avgn.utils.cuda_variable import cuda_variable
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
import torch
import librosa


import IPython.display as ipd


# %%
# Get chunks
def get_chunks(path, hparams):
    # mel basis
    mel_basis = build_mel_basis(hparams, hparams.sr, hparams.sr)
    # load file
    x_s, _ = prepare_wav(wav_loc=path, hparams=hparams, debug=False)
    # Segmentation params
    min_level_db_floor = -40
    db_delta = 2
    silence_threshold = 0.1
    min_silence_for_spec = 0.05
    max_vocal_for_spec = 1.0
    min_syllable_length_s = 0.05
    #defaults
    #min_level_db_floor = -30
    #db_delta = 5
    #silence_threshold = 0.01
    #min_silence_for_spec = 0.05
    #max_vocal_for_spec = 1.0,
    #min_syllable_length_s = 0.05

    # segment
    results = dynamic_threshold_segmentation(
        x_s,
        hparams.sr,
        n_fft=hparams.n_fft,
        hop_length=hparams.hop_length_samples,
        win_length=hparams.win_length_samples,
        min_level_db_floor=min_level_db_floor,
        db_delta=db_delta,
        ref_level_db=hparams.ref_level_db,
        pre=hparams.preemphasis,
        min_silence_for_spec=min_silence_for_spec,
        max_vocal_for_spec=max_vocal_for_spec,
        silence_threshold=silence_threshold,
        verbose=False,
        min_syllable_length_s=min_syllable_length_s,
        spectral_range=[hparams.mel_lower_edge_hertz, hparams.mel_upper_edge_hertz],
    )
    if results is None:
        print('Cannot segment the input file')
        return
    # chunks
    start_times = results["onsets"]
    end_times = results["offsets"]
    chunks_mS = []
    start_samples = []
    end_samples = []
    for start_time, end_time in zip(start_times, end_times):
        start_sample = int(start_time * hparams.sr)
        end_sample = int(end_time * hparams.sr)
        syl = x_s[start_sample:end_sample]

        # To avoid mistakes, reproduce the whole preprocessing pipeline, even (here useless) int casting
        _, mS, _ = process_syllable(syl, hparams, mel_basis=mel_basis, debug=False)
        if mS is None:
            continue
        mS_int = (mS * 255).astype('uint8')
        sample = SpectroDataset.process_mSp(mS_int)

        chunks_mS.append(sample)
        start_samples.append(start_sample)
        end_samples.append(end_sample)
    return x_s, chunks_mS, start_samples, end_samples


# %%
# Get chunks
def get_contam_chunks(path, source_start_samples, source_end_samples, hparams):
    # mel basis
    mel_basis = build_mel_basis(hparams, hparams.sr, hparams.sr)
    # load file
    x_s, _ = prepare_wav(wav_loc=path, hparams=hparams, debug=False)
    # Segmentation params
    #min_level_db_floor = -30
    #db_delta = 6
    #silence_threshold = 0.1
    #min_silence_for_spec = 0.05
    #max_vocal_for_spec = 1.0,
    #min_syllable_length_s = 0.1
    # segment
#    results = dynamic_threshold_segmentation(
#        x_s,
#        hparams.sr,
#        n_fft=hparams.n_fft,
#        hop_length=hparams.hop_length_samples,
#        win_length=hparams.win_length_samples,
#        min_level_db_floor=min_level_db_floor,
#        db_delta=db_delta,
#        ref_level_db=hparams.ref_level_db,
#        pre=hparams.preemphasis,
#        min_silence_for_spec=min_silence_for_spec,
#        max_vocal_for_spec=max_vocal_for_spec,
#        silence_threshold=silence_threshold,
#        verbose=False,
#        min_syllable_length_s=min_syllable_length_s,
#        spectral_range=[hparams.mel_lower_edge_hertz, hparams.mel_upper_edge_hertz],
#    )
#    if results is None:
#        print('Cannot segment the input file')
#        return
    # chunks
    start_times = [i / hparams.sr for i in source_start_samples]
    end_times = [i / hparams.sr for i in source_end_samples]
    chunks_mS = []
    start_samples = []
    end_samples = []
    for start_time, end_time in zip(start_times, end_times):
        start_sample = int(start_time * hparams.sr)
        end_sample = int(end_time * hparams.sr)
        syl = x_s[start_sample:end_sample]

        # To avoid mistakes, reproduce the whole preprocessing pipeline, even (here useless) int casting
        _, mS, _ = process_syllable(syl, hparams, mel_basis=mel_basis, debug=False)
        if mS is None:
            continue
        mS_int = (mS * 255).astype('uint8')
        sample = SpectroDataset.process_mSp(mS_int)

        chunks_mS.append(sample)
        start_samples.append(start_sample)
        end_samples.append(end_sample)
    return x_s, chunks_mS, start_samples, end_samples


# %%


# %% [markdown]
# ## Set model and material

# %%
#config_path = "models/VAE_voizo_2021-04-06_15-23-04/config.py"
#loading_epoch = 1750
#source_path = '/home/leo/Code/birds_latent_generation/data/raw/voizo_chunks/Nigthingale/XCcommonNightingale-Denoised/Nightingale1_0_0.wav'
#contamination_path = '/home/leo/Code/birds_latent_generation/data'\
#    '/raw/voizo_chunks/Corvus/XCcorvus-Denoised/Kraai_BieslNp_120312-07xc_0_0.wav'
#contamination_parameters = {
#    'p_contamination': 0.5,
#}
#method = 'linear'



config_path = "models/VAE_ra-mo-ni-ro_2021-05-05_23-04-21/config.py"
loading_epoch = 5400
#config_path = "models/VAE_ra-mo-ni-ro_2021-04-19_22-18-08/config.py"
#loading_epoch = 400
#contamination_path = '/home/syrinx/birds_latent_generation/data/raw/ra-mo-ni-ro_chunks/allraven001-contam.wav'
#contamination_path = '/home/syrinx/birds_latent_generation/data/raw/ra-mo-ni-ro_chunks/mediumnightingale001.wav'
#source_path = '/home/syrinx/birds_latent_generation/data/raw/ra-mo-ni-ro_chunks/mediumrobin001-contam.wav'
#source_path = '/home/syrinx/birds_latent_generation/data/raw/ra-mo-ni-ro_chunks/mediumnightingale001.wav'
#contamination_path = '/home/syrinx/birds_latent_generation/data/raw/ra-mo-ni-ro_chunks/mediumnightingale001.wav'
#contamination_path = '/home/syrinx/birds_latent_generation/data/raw/ra-mo-ni-ro_chunks/allraven001-contam.wav'
source_path = '/home/syrinx/birds_latent_generation/data/audioguide/shortmediummocking.wav'
contamination_path = '/home/syrinx/birds_latent_generation/data/audioguide/shortmediummocking-raven-AG.wav'
#source_path = '/home/syrinx/birds_latent_generation/data/audioguide/mediumrobin-short.wav'
#contamination_path = '/home/syrinx/birds_latent_generation/data/audioguide/robin-nightingale-audioguide.wav'


contamination_parameters = {
    'p_contamination': 0.9, 'contam_degree' : 1.,
}
method = 'linear'
#method = 'constant_radius'

# %% [markdown]
# ## Generate contamination

# %%
# load model
model, _, _, _, hparams, _, _, config_path = get_model_and_dataset(
    config=config_path, loading_epoch=loading_epoch)

# load files
source = {}
source['path'] = source_path
waveform, chunks, start_samples, end_samples = get_chunks(path=source['path'], hparams=hparams)
source['waveform'] = waveform
source['chunks'] = chunks
source['start_samples'] = start_samples
source['end_samples'] = end_samples
contamination = {}
contamination['path'] = contamination_path
#waveform, chunks, start_samples, end_samples = get_chunks(path=contamination['path'], hparams=hparams)

waveform, chunks, start_samples, end_samples = get_contam_chunks(path=contamination['path'], source_start_samples=source['start_samples'], source_end_samples=source['end_samples'], hparams=hparams)
contamination['waveform'] = waveform
contamination['chunks'] = chunks
contamination['start_samples'] = start_samples
contamination['end_samples'] = end_samples

# Choose which samples to contaminate and by which degree
contamination_indices = []
contamination_degrees = []
time_shift = 0
xs = []
ys = []
p_contamination = contamination_parameters['p_contamination']
contamination_constant = contamination_parameters['contam_degree']
for index, chunk in enumerate(source['chunks']):
    if random.random() < p_contamination:
        contamination_indices.append(index)
        #contamination_degrees = 0.9
        #contamination_degrees.append(random.random())
        contamination_degrees.append(contamination_constant)
        xs.append(chunk)
        # choose (randomly?) a contaminating syllable
#        ys.append(random.choice(contamination['chunks']))
        # choose the contamination chunk with the same index as the source chunk
        # (because they're aligned with audioguide)
        ys.append(contamination['chunks'][index])
xs_cuda = cuda_variable(torch.tensor(np.stack(xs)))
ys_cuda = cuda_variable(torch.tensor(np.stack(ys)))

# Encode
mu, logvar = model.encode(xs_cuda)
x_z = model.reparameterize(mu, logvar)
mu, logvar = model.encode(ys_cuda)
y_z = model.reparameterize(mu, logvar)
z_out = torch.zeros_like(x_z)

# Contaminate
for batch_ind, t in enumerate(contamination_degrees):
    if method == 'linear':
        z_out[batch_ind] = x_z[batch_ind] * (1 - t) + y_z[batch_ind] * t
    elif method == 'constant_radius':
        z_out[batch_ind] = constant_radius_interpolation(x_z[batch_ind], y_z[batch_ind], t)
# Decode z
x_recon = model.decode(z_out).cpu().detach().numpy()


# Replace contamined samples in original wave
#out_wave = source['waveform'].copy()
#mel_basis = build_mel_basis(hparams, hparams.sr, hparams.sr)
#mel_inversion_basis = build_mel_inversion_basis(mel_basis)
#for batch_index, contamination_index in enumerate(contamination_indices):
#    new_chunk = x_recon[batch_index, 0]
#    new_audio = inv_spectrogram_sp(
#        new_chunk,
#        n_fft=hparams.n_fft,
#        win_length=hparams.win_length_samples,
#        hop_length=hparams.hop_length_samples,
#        ref_level_db=hparams.ref_level_db,
#        power=hparams.power,
#        mel_inversion_basis=mel_inversion_basis
#    )
#    start_sample = source['start_samples'][contamination_index]
#    end_sample = source['end_samples'][contamination_index]
#    length_sample = end_sample - start_sample
#    # FAIRE UN FADE ICI
#    out_wave[start_sample:end_sample] = new_audio[:length_sample]

#####################################################################################################
#####################################################################################################
# Replace contamined samples in original wave
out_wave = source['waveform'].copy()
mel_basis = build_mel_basis(hparams, hparams.sr, hparams.sr)
mel_inversion_basis = build_mel_inversion_basis(mel_basis)
for batch_index, contamination_index in enumerate(contamination_indices):
    new_chunk = x_recon[batch_index, 0]
    s_unnorm = _denormalize(
        new_chunk, min_db=_min_level_db(), max_db=hparams.ref_level_db)
    s_amplitude = _db_to_amplitude(s_unnorm + hparams.ref_level_db)
    s_linear = _mel_to_linear(
        s_amplitude, _mel_inverse_basis=mel_inversion_basis)**(1 / hparams.power)

    # Calculer la dimension de la syllabe générée (rms energy)
    rms_energy = librosa.feature.rms(S=s_linear, frame_length=hparams.win_length_samples,
                                     hop_length=hparams.hop_length_samples)
    rms_energy = rms_energy[0, :]
    rms_energy_norm = rms_energy / (rms_energy.max() + 1e-12)
    non_silence_indices = np.where(rms_energy_norm > 0.05)[0]
    start_index = non_silence_indices.min()
    end_index = non_silence_indices.max()
    s_chunk = s_linear[:, start_index:end_index]

    # griffin-lim
    x_grif = griffinlim_sp(s_chunk, n_fft=hparams.n_fft,
                           win_length=hparams.win_length_samples, hop_length=hparams.hop_length_samples)

    # Match gain
    start_sample = source['start_samples'][contamination_index]
    end_sample = source['end_samples'][contamination_index]
    y = out_wave[start_sample:end_sample]
    time_shift += len(x_grif) - len(y)

    start_sample_shifted = start_sample + time_shift
    end_sample_shifted = end_sample + time_shift


    # gain_source = np.abs(y).max()
    # gain_target = np.abs(x_grif).max()
    if len(y) >> 0:
        gain_source = librosa.feature.rms(y).max()
    else:
        gain_source = 0.
    gain_target = librosa.feature.rms(x_grif).max()
    #print(x_grif * gain_source / gain_target)
    if gain_target > gain_source:
        x_norm = x_grif * gain_source / gain_target
    else:
        x_norm = x_grif



    # Insérer
#    out_wave = np.concatenate(
#        (out_wave[:start_sample], x_norm, out_wave[end_sample:])
#    )
    out_wave = np.concatenate(
        (out_wave[:start_sample_shifted], x_norm, out_wave[end_sample_shifted:])
    )

#####################################################################################################
#####################################################################################################


ipd.display(
        #ipd.Audio(source['waveform'], rate=hparams.sr),
        #ipd.Audio(contamination['waveform'], rate=hparams.sr),
        ipd.Audio(out_wave, rate=hparams.sr),
)


# %%
contamination['chunks'][4]

# %% [markdown]
# ## Check segmentation

# %%
# load model
#model, _, _, _, hparams, _, _, config_path = get_model_and_dataset(
#    config=config_path, loading_epoch=loading_epoch)

# load files
source = {}
source['path'] = source_path
waveform, chunks, start_samples, end_samples = get_chunks(path=source['path'], hparams=hparams)
source['waveform'] = waveform
source['chunks'] = chunks
source['start_samples'] = start_samples
source['end_samples'] = end_samples


# %%
#from vocalseg.dynamic_thresholding import plot_segmented_spec, plot_segmentations

for i in range(len(source['start_samples'])):
    print("%s\t%s" % (source['start_samples'][i]/hparams.sr, source['end_samples'][i]/hparams.sr))


# %%



# %%



# %%



# %%



