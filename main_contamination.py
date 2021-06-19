# Contamination
from avgn.utils.get_chunks import get_chunks
import math
import shutil
import librosa
import torch
import soundfile as sf
import numpy as np
from avgn.utils.cuda_variable import cuda_variable
from avgn.pytorch.generate.interpolation import constant_radius_interpolation
import random
import os
from avgn.pytorch.getters import get_model_and_dataset
from avgn.signalprocessing.spectrogramming_scipy import _db_to_amplitude, _denormalize, _mel_to_linear, _min_level_db, \
    build_mel_basis, build_mel_inversion_basis, griffinlim_sp


# Set model and material
config_path = "models/VAE_MMD_du_ra_mo_ni_ro_2021-06-13_22-03-24/config.py"
loading_epoch = 1080
source_path = '/home/syrinx/birds_latent_generation/data/audioguide/shortmediummocking.wav'
contamination_path = '/home/syrinx/birds_latent_generation/data/audioguide/shortmediummocking-raven-AG.wav'
max_batch_size = 4

contamination_parameters = {
    'p_contamination': 1.0, 'contam_degree': 1.0,
}
method = 'linear'
# method = 'constant_radius'

# load model
model, _, _, _, hparams, _, _, config_path = get_model_and_dataset(
    config=config_path, loading_epoch=loading_epoch)

# load files
source = {}
source['path'] = source_path
waveform, chunks, start_samples, end_samples = get_chunks(
    path=source['path'], hparams=hparams)
source['waveform'] = waveform
source['chunks'] = chunks
source['start_samples'] = start_samples
source['end_samples'] = end_samples
contamination = {}
contamination['path'] = contamination_path
waveform, chunks, start_samples, end_samples = get_chunks(
    path=contamination['path'], hparams=hparams)
# waveform, chunks, start_samples, end_samples = \
#     get_contam_chunks(path=contamination['path'], source_start_samples=source['start_samples'],
#                       source_end_samples=source['end_samples'], hparams=hparams)
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
        # contamination_degrees = 0.9
        # contamination_degrees.append(random.random())
        contamination_degrees.append(contamination_constant)
        xs.append(chunk)

        # Select contamination
        # 1/ choose (randomly?) a contaminating syllable
        # ys.append(random.choice(contamination['chunks']))
        # 2/ choose the contamination chunk with the same index as the source chunk
        # (because they're aligned with audioguide)
        # ys.append(contamination['chunks'][index])
        # 3/ Choose the closest segment in ys
        start_sample_source = source['start_samples'][index]
        smallest_distance = math.inf
        index_contam_selected = None
        for index_contam, start_sample_contam in enumerate(contamination['start_samples']):
            this_distance = abs(start_sample_source - start_sample_contam)
            if this_distance < smallest_distance:
                index_contam_selected = index_contam
                smallest_distance = this_distance
        ys.append(contamination['chunks'][index_contam_selected])
xs_cuda = cuda_variable(torch.tensor(np.stack(xs)))
ys_cuda = cuda_variable(torch.tensor(np.stack(ys)))
print(f'Encode {len(xs_cuda)} source syllables and {len(ys_cuda)} contam syllables')

# Encode
num_chunks_source = len(xs_cuda) // max_batch_size + 1
x_recon_l = []
batch_ind = 0
for xs_b, ys_b in zip(torch.chunk(xs_cuda, num_chunks_source, dim=0), torch.chunk(ys_cuda, num_chunks_source, dim=0)):
    x_z = model.reparameterize(*model.encode(xs_b))
    y_z = model.reparameterize(*model.encode(ys_b))
    z_out = torch.zeros_like(x_z)
    # Contaminate
    for b_ind in range(len(xs_b)):
        t = contamination_degrees[batch_ind]
        if method == 'linear':
            z_out[b_ind] = x_z[b_ind] * (1 - t) + y_z[b_ind] * t
        elif method == 'constant_radius':
            z_out[b_ind] = constant_radius_interpolation(
                x_z[np.newaxis, b_ind], y_z[np.newaxis, b_ind], t)
        batch_ind += 1
    # Decode z
    x_recon_l.append(model.decode(z_out).cpu().detach().numpy())
x_recon = np.concatenate(x_recon_l)

# Test encode and decode source chunks
mel_basis = build_mel_basis(hparams, hparams.sr, hparams.sr)
mel_inversion_basis = build_mel_inversion_basis(mel_basis)
recon_source = model.decode(x_z).cpu().detach().numpy()
recon_target = model.decode(y_z).cpu().detach().numpy()
if os.path.isdir('dump/reconstructions'):
    shutil.rmtree('dump/reconstructions')
os.makedirs('dump/reconstructions')
for batch_index in range(len(recon_source)):
    # Original
    new_chunk = xs_cuda[batch_index, 0].cpu().detach().numpy()
    s_unnorm = _denormalize(
        new_chunk, min_db=_min_level_db(), max_db=hparams.ref_level_db)
    s_amplitude = _db_to_amplitude(s_unnorm + hparams.ref_level_db)
    s_linear = _mel_to_linear(
        s_amplitude, _mel_inverse_basis=mel_inversion_basis)**(1 / hparams.power)
    # griffin-lim
    x_grif = griffinlim_sp(s_linear, n_fft=hparams.n_fft,
                           win_length=hparams.win_length_samples, hop_length=hparams.hop_length_samples)
    sf.write(f'dump/reconstructions/{batch_index}_source_original.wav',
             x_grif, samplerate=hparams.sr)
    # Reconstruction
    new_chunk = recon_source[batch_index, 0]
    s_unnorm = _denormalize(
        new_chunk, min_db=_min_level_db(), max_db=hparams.ref_level_db)
    s_amplitude = _db_to_amplitude(s_unnorm + hparams.ref_level_db)
    s_linear = _mel_to_linear(
        s_amplitude, _mel_inverse_basis=mel_inversion_basis)**(1 / hparams.power)
    # griffin-lim
    x_grif = griffinlim_sp(s_linear, n_fft=hparams.n_fft,
                           win_length=hparams.win_length_samples, hop_length=hparams.hop_length_samples)
    sf.write(f'dump/reconstructions/{batch_index}_source_reconstruction.wav',
             x_grif, samplerate=hparams.sr)
    if batch_index > 50:
        break
for batch_index in range(len(recon_target)):
    # Original
    new_chunk = ys_cuda[batch_index, 0].cpu().detach().numpy()
    s_unnorm = _denormalize(
        new_chunk, min_db=_min_level_db(), max_db=hparams.ref_level_db)
    s_amplitude = _db_to_amplitude(s_unnorm + hparams.ref_level_db)
    s_linear = _mel_to_linear(
        s_amplitude, _mel_inverse_basis=mel_inversion_basis)**(1 / hparams.power)
    # griffin-lim
    x_grif = griffinlim_sp(s_linear, n_fft=hparams.n_fft,
                           win_length=hparams.win_length_samples, hop_length=hparams.hop_length_samples)
    sf.write(f'dump/reconstructions/{batch_index}_target_original.wav',
             x_grif, samplerate=hparams.sr)
    # Reconstruction
    new_chunk = recon_target[batch_index, 0]
    s_unnorm = _denormalize(
        new_chunk, min_db=_min_level_db(), max_db=hparams.ref_level_db)
    s_amplitude = _db_to_amplitude(s_unnorm + hparams.ref_level_db)
    s_linear = _mel_to_linear(
        s_amplitude, _mel_inverse_basis=mel_inversion_basis)**(1 / hparams.power)
    # griffin-lim
    x_grif = griffinlim_sp(s_linear, n_fft=hparams.n_fft,
                           win_length=hparams.win_length_samples, hop_length=hparams.hop_length_samples)
    sf.write(f'dump/reconstructions/{batch_index}_target_reconstruction.wav',
             x_grif, samplerate=hparams.sr)
    if batch_index > 50:
        break

#####################################################################################################
#####################################################################################################
# Replace contamined samples in original wave
out_wave = source['waveform'].copy()
mel_basis = build_mel_basis(hparams, hparams.sr, hparams.sr)
mel_inversion_basis = build_mel_inversion_basis(mel_basis)
if os.path.isdir('dump/syllables_gen'):
    shutil.rmtree('dump/syllables_gen')
os.makedirs('dump/syllables_gen')
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

    if batch_index < 10:
        sf.write(
            f'dump/syllables_gen/{batch_index}_syll.wav', x_grif, samplerate=hparams.sr)

    # Match gain
    start_sample = source['start_samples'][contamination_index]
    end_sample = source['end_samples'][contamination_index]
    y = source['waveform'][start_sample:end_sample]
    if len(y) > 0:
        gain_source = librosa.feature.rms(y).max()
    else:
        gain_source = 0.
    gain_target = librosa.feature.rms(x_grif).max()
    if gain_target > gain_source:
        x_norm = x_grif * gain_source / gain_target
    else:
        x_norm = x_grif

    if batch_index < 10:
        sf.write(
            f'dump/syllables_gen/{batch_index}_syllNorm.wav', x_norm, samplerate=hparams.sr)

    # Fade
    def fade(sig, start_end, max_len_sec):
        fade_len = int(min(len(sig), max_len_sec * hparams.sr))
        fade = (np.linspace(0, fade_len, fade_len) / fade_len)**2
        if start_end == 'start':
            sig[-fade_len:] = sig[-fade_len:] * fade
        elif start_end == 'end':
            fade = 1 - fade
            sig[:fade_len] = sig[-fade_len:] * fade
        return sig

    if batch_index < 10:
        sf.write(
            f'dump/syllables_gen/{batch_index}_syllFade.wav', fade(
                x_norm, start_end='end', max_len_sec=0.005), samplerate=hparams.sr)

    # Insérer
    start_sample_shifted = start_sample + time_shift
    end_sample_shifted = end_sample + time_shift
    if batch_index < 10:
        sf.write(f'dump/syllables_gen/{batch_index}_syllRemoved.wav', out_wave[start_sample_shifted:end_sample_shifted],
                 samplerate=hparams.sr)
    out_wave = np.concatenate(
        (fade(out_wave[:start_sample_shifted], start_end='end', max_len_sec=0.005),
         fade(x_norm, start_end='end', max_len_sec=0.005),
         fade(out_wave[end_sample_shifted:], start_end='end', max_len_sec=0.005))
    )
    time_shift += len(x_norm) - len(y)

sf.write('dump/contamination.wav', out_wave, samplerate=hparams.sr)


# ipd.display(
#     # ipd.Audio(source['waveform'], rate=hparams.sr),
#     # ipd.Audio(contamination['waveform'], rate=hparams.sr),
#     ipd.Audio(out_wave, rate=hparams.sr),
# )
