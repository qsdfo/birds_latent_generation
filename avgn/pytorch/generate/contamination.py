from avgn.pytorch.dataset.spectro_dataset import SpectroDataset
from avgn.pytorch.generate.interpolation import constant_radius_interpolation
import random
from main_spectrogramming import process_syllable
from avgn.signalprocessing.dynamic_thresholding_scipy import dynamic_threshold_segmentation
from avgn.signalprocessing.create_spectrogram_dataset import prepare_wav
import os
from avgn.pytorch.getters import get_model_and_dataset
from avgn.signalprocessing.spectrogramming_scipy import build_mel_basis, build_mel_inversion_basis, inv_spectrogram_sp
from avgn.utils.cuda_variable import cuda_variable
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
import torch


def main(config_path, loading_epoch, source_path, contamination_path, contamination_parameters, method):
    # load model
    model, _, _, _, hparams, _, _, config_path = get_model_and_dataset(
        config=config_path, loading_epoch=loading_epoch)

    # set savepath
    savepath = f'{model.model_dir}/plots/contaminations'
    if not os.path.isdir(savepath):
        os.makedirs(savepath)

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
    contamination['waveform'] = waveform
    contamination['chunks'] = chunks
    contamination['start_samples'] = start_samples
    contamination['end_samples'] = end_samples

    # Choose which samples to contaminate and by which degree
    contamination_indices = []
    contamination_degrees = []
    xs = []
    ys = []
    p_contamination = contamination_parameters['p_contamination']
    for index, chunk in enumerate(source['chunks']):
        if random.random() < p_contamination:
            contamination_indices.append(index)
            contamination_degrees.append(random.random())
            xs.append(chunk)
            # choose (randomly?) a contaminating syllable
            ys.append(random.choice(contamination['chunks']))
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
            z_out[batch_ind] = constant_radius_interpolation(
                x_z[batch_ind], y_z[batch_ind], t)
    # Decode z
    x_recon = model.decode(z_out).cpu().detach().numpy()

    # Replace contamined samples in original wave
    out_wave = source['waveform']
    mel_basis = build_mel_basis(hparams, hparams.sr, hparams.sr)
    mel_inversion_basis = build_mel_inversion_basis(mel_basis)
    for batch_index, contamination_index in enumerate(contamination_indices):
        new_chunk = x_recon[batch_index, 0]
        new_audio = inv_spectrogram_sp(
            new_chunk,
            n_fft=hparams.n_fft,
            win_length=hparams.win_length_samples,
            hop_length=hparams.hop_length_samples,
            ref_level_db=hparams.ref_level_db,
            power=hparams.power,
            mel_inversion_basis=mel_inversion_basis
        )
        start_sample = source['start_samples'][contamination_index]
        end_sample = source['end_samples'][contamination_index]
        length_sample = end_sample - start_sample
        # FAIRE UN FADE ICI
        out_wave[start_sample:end_sample] = new_audio[:length_sample]
    sf.write(f'{savepath}/contamination.wav', out_wave, samplerate=hparams.sr)

# Get chunks


def get_chunks(path, hparams):
    # mel basis
    mel_basis = build_mel_basis(hparams, hparams.sr, hparams.sr)
    # load file
    x_s, _ = prepare_wav(wav_loc=path, hparams=hparams, debug=False)
    # Segmentation params
    min_level_db_floor = -30
    db_delta = 5
    silence_threshold = 0.01
    min_silence_for_spec = 0.05
    max_vocal_for_spec = 1.0,
    min_syllable_length_s = 0.05
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
        verbose=True,
        min_syllable_length_s=min_syllable_length_s,
        spectral_range=[hparams.mel_lower_edge_hertz,
                        hparams.mel_upper_edge_hertz],
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
        _, mS, _ = process_syllable(
            syl, hparams, mel_basis=mel_basis, debug=False)
        if mS is None:
            continue
        mS_int = (mS * 255).astype('uint8')
        sample = SpectroDataset.process_mSp(mS_int, chunk_len_win=hparams['chunk_len_win'],
                                            num_mel_bins=hparams['num_mel_bins'])

        chunks_mS.append(sample)
        start_samples.append(start_sample)
        end_samples.append(end_sample)
    return x_s, chunks_mS, start_samples, end_samples


if __name__ == '__main__':
    config_path = "models/VAE_voizo_2021-04-05_15-31-14/config.py"
    loading_epoch = 40
    source_path = '/home/leo/Code/birds_latent_generation/data' \
        '/raw/voizo_chunks/Nigthingale/XCcommonNightingale-Denoised/Nightingale1_0_0.wav'
    contamination_path = '/home/leo/Code/birds_latent_generation/data'\
        '/raw/voizo_chunks/Corvus/XCcorvus-Denoised/Kraai_BieslNp_120312-07xc_0_0.wav'
    contamination_parameters = {
        'p_contamination': 0.5,
    }
    method = 'linear'

    main(config_path=config_path,
         loading_epoch=loading_epoch,
         source_path=source_path,
         contamination_path=contamination_path,
         contamination_parameters=contamination_parameters,
         method=method)
