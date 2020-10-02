"""
Used to test the preprocessing stack.
Not necessary for training or generating
"""

import itertools
import os
import pickle
import shutil

import librosa
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from avgn.signalprocessing.create_spectrogram_dataset import prepare_wav
from avgn.signalprocessing.spectrogramming import build_mel_basis, build_mel_inversion_basis, inv_spectrogram_librosa, \
    spectrogram_librosa, _denormalize
from avgn.utils.hparams import HParams
from avgn.utils.paths import DATA_DIR


def single_file_test(num_mel_bins, n_fft, mel_lower_edge_hertz, mel_upper_edge_hertz, hop_length_ms, win_length_ms_l,
                     power):
    hparams = HParams(
        sr=44100,
        num_mel_bins=num_mel_bins,
        n_fft=n_fft,
        mel_lower_edge_hertz=mel_lower_edge_hertz,
        mel_upper_edge_hertz=mel_upper_edge_hertz,
        power=power,  # for spectral inversion
        butter_lowcut=mel_lower_edge_hertz,
        butter_highcut=mel_upper_edge_hertz,
        ref_level_db=40,  # 20
        min_level_db=-90,  # 60
        mask_spec=False,
        win_length_ms=win_length_ms_l,
        hop_length_ms=hop_length_ms,
        mask_spec_kwargs={"spec_thresh": 0.9, "offset": 1e-10},
        n_jobs=1,
        verbosity=1,
        reduce_noise=True,
        noise_reduce_kwargs={"n_std_thresh": 2.0, "prop_decrease": 0.8},
        griffin_lim_iters=50
    )
    suffix = hparams.__repr__()

    dump_folder = DATA_DIR / 'dump' / f'{suffix}'
    if os.path.isdir(dump_folder):
        shutil.rmtree(dump_folder)
    os.makedirs(dump_folder)

    data_dir = '/home/leo/Recherche/Code/birds_project/birds_latent_generation/data'
    wav_loc = f'{data_dir}/raw/bird-db/CATH/CATH-CP1/wavs/2009-03-21_08-27-00-000000.wav'

    #  Read wave
    data, _ = prepare_wav(wav_loc, hparams, dump_folder, debug=True)
    data = data[:hparams.sr * 15]
    # create spec
    mel_basis = build_mel_basis(hparams, hparams.sr, hparams.sr)
    mel_inversion_basis = build_mel_inversion_basis(mel_basis)
    melspec, debug_info = spectrogram_librosa(data, hparams, _mel_basis=mel_basis, debug=True)

    from avgn.signalprocessing.spectrogramming import griffinlim_librosa, _mel_to_linear

    ind = 'testfile'
    librosa.output.write_wav(f'{dump_folder}/{ind}_syllable.wav', data,
                             sr=hparams.sr, norm=True)

    #  preemphasis y
    librosa.output.write_wav(f'{dump_folder}/{ind}_preemphasis_y.wav', debug_info['preemphasis_y'],
                             sr=hparams.sr, norm=True)

    # test librosa stft and istft
    S = librosa.core.stft(debug_info['preemphasis_y'])
    plt.clf()
    plt.matshow(np.abs(S)[:, :100], origin="lower")
    plt.savefig(f'{dump_folder}/{ind}_original_S_abs.pdf')
    plt.close()
    plt.clf()
    plt.matshow(np.angle(S)[:, :100], origin="lower")
    plt.savefig(f'{dump_folder}/{ind}_original_S_angle.pdf')
    plt.close()
    plt.clf()
    plt.matshow(np.imag(S)[:, :100], origin="lower")
    plt.savefig(f'{dump_folder}/{ind}_original_S_imag.pdf')
    plt.close()
    plt.clf()
    plt.matshow(np.real(S)[:, :100], origin="lower")
    plt.savefig(f'{dump_folder}/{ind}_original_S_real.pdf')
    plt.close()
    y_istft_librosa = librosa.core.istft(S)
    librosa.output.write_wav(f'{dump_folder}/{ind}_librosa_istft.wav', y_istft_librosa, sr=hparams.sr, norm=True)

    # S_abs
    plt.clf()
    plt.matshow(debug_info['S_abs'][:, :100], origin="lower")
    plt.savefig(f'{dump_folder}/{ind}_S_abs.pdf')
    plt.close()
    S_abs_inv = griffinlim_librosa(debug_info['S'], hparams.sr, hparams)
    # S_angle_griffinlim = np.angle(librosa.core.stft(S_abs_inv))
    # plt.clf()
    # plt.matshow(S_angle_griffinlim, origin="lower")
    # plt.savefig(f'{dump_folder}/{ind}_S_angle.pdf')
    # plt.close()
    librosa.output.write_wav(f'{dump_folder}/{ind}_S_abs.wav', S_abs_inv, sr=hparams.sr, norm=True)

    # mel
    plt.clf()
    plt.matshow(debug_info['mel'][:, :100], origin="lower")
    plt.savefig(f'{dump_folder}/{ind}_mel.pdf')
    plt.close()
    mel_inv = griffinlim_librosa(
        _mel_to_linear(debug_info['mel'], _mel_inverse_basis=mel_inversion_basis), 
        hparams.sr, hparams)
    librosa.output.write_wav(f'{dump_folder}/{ind}_mel.wav', mel_inv, sr=hparams.sr, norm=True)

    # mel_db
    plt.clf()
    plt.matshow(debug_info['mel_db'][:, :100], origin="lower")
    plt.savefig(f'{dump_folder}/{ind}_mel_db.pdf')
    plt.close()
    aa_ = debug_info['mel_db'] + hparams.ref_level_db
    bb_ = librosa.db_to_amplitude(aa_)
    cc_ = _mel_to_linear(bb_, _mel_inverse_basis=mel_inversion_basis)
    mel_db_inv = griffinlim_librosa(cc_, hparams.sr, hparams)
    librosa.output.write_wav(f'{dump_folder}/{ind}_mel_db.wav', mel_db_inv, sr=hparams.sr, norm=True)

    # mel_db_norm
    plt.clf()
    plt.matshow(debug_info['mel_db_norm'][:, :100], origin="lower")
    plt.savefig(f'{dump_folder}/{ind}_mel_db_norm.pdf')
    plt.close()
    # zz = _denormalize(debug_info['mel_db_norm'], hparams)
    # aa = zz + hparams.ref_level_db
    # bb = librosa.db_to_amplitude(aa)
    # cc = _mel_to_linear(bb, _mel_inverse_basis=mel_inversion_basis)
    # mel_db_norm_inv = griffinlim_librosa(cc, hparams.sr, hparams)
    mel_db_norm_inv = inv_spectrogram_librosa(debug_info['mel_db_norm'], hparams.sr, hparams,
                                              mel_inversion_basis=mel_inversion_basis)
    librosa.output.write_wav(f'{dump_folder}/{ind}_mel_db_norm.wav', mel_db_norm_inv, sr=hparams.sr, norm=True)


def main(dataset_name):
    ##################################################################################
    print('##### Dataset')

    # Choose a preprocessing pipeline
    # win_length_ms = None
    # hop_length_ms = 10
    # n_fft = 2048
    # num_mel_bins = 128
    # mel_lower_edge_hertz = 500
    # mel_upper_edge_hertz = 12000
    # suffix_preprocessing = f'wl{win_length_ms}_' \
    #                         f'hl{hop_length_ms}_' \
    #                         f'nfft{n_fft}_' \
    #                         f'melb{num_mel_bins}_' \
    #                         f'mell{mel_lower_edge_hertz}_' \
    #                         f'melh{mel_upper_edge_hertz}'
    suffix_preprocessing = 'wlNone_hl5_nfft4096_melb256_mell500_melh20000'

    df_loc = DATA_DIR / 'syllable_dfs' / dataset_name / f'data_{suffix_preprocessing}.pickle'
    syllable_df = pd.read_pickle(df_loc)

    hparams_loc = DATA_DIR / 'syllable_dfs' / dataset_name / f'hparams_{suffix_preprocessing}.pickle'
    with open(hparams_loc, 'rb') as ff:
        hparams = pickle.load(ff)

    dump_folder = DATA_DIR / 'dump' / 'reconstruction'
    for ind in range(len(syllable_df)):
        if ind > 50:
            break
        spec = syllable_df.spectrogram[ind] / 255.
        audio = syllable_df.audio[ind]

        # plt spectrogram
        plt.clf()
        plt.matshow(spec, origin="lower")
        plt.savefig(f'{dump_folder}/{ind}_spectro.pdf')
        plt.close()

        # melSpec to audio
        mel_basis = build_mel_basis(hparams, hparams.sr, hparams.sr)
        mel_inversion_basis = build_mel_inversion_basis(mel_basis)
        audio_reconstruct = inv_spectrogram_librosa(spec, hparams.sr, hparams, mel_inversion_basis=mel_inversion_basis)

        # save and plot audio original and reconstructed
        librosa.output.write_wav(f'{dump_folder}/{ind}_reconstruct.wav', audio_reconstruct, sr=hparams.sr, norm=True)
        librosa.output.write_wav(f'{dump_folder}/{ind}_original.wav', audio, sr=hparams.sr, norm=True)


def plot_reconstruction(model, dataloader, device, savepath):
    # Forward pass
    model.eval()
    for _, data in enumerate(dataloader):
        data_cuda = data.to(device)
        x_recon = model.reconstruct(data_cuda).cpu().detach().numpy()
        break
    # Plot
    dims = x_recon.shape[2:]
    num_examples = x_recon.shape[0]
    fig, axes = plt.subplots(nrows=2, ncols=num_examples)
    for i in range(num_examples):
        # show the image
        axes[0, i].matshow(data[i].reshape(dims), origin="lower")
        axes[1, i].matshow(x_recon[i].reshape(dims), origin="lower")
    for ax in fig.get_axes():
        ax.set_xticks([])
        ax.set_yticks([])
    plt.savefig(savepath)
    plt.close()
    plt.clf()


def plot_generation(model, num_examples, savepath):
    # forward pass
    model.eval()
    gen = model.generate(batch_dim=num_examples).cpu().detach().numpy()
    # plot
    dims = gen.shape[2:]
    fig, axes = plt.subplots(ncols=num_examples)
    for i in range(num_examples):
        # show the image
        axes[i].matshow(gen[i].reshape(dims), origin="lower")
    for ax in fig.get_axes():
        ax.set_xticks([])
        ax.set_yticks([])
    plt.savefig(savepath)
    plt.close()
    plt.clf()


def epoch(model, optimizer, dataloader, num_batches, training, device):
    if training:
        model.train()
    else:
        model.eval()
    losses = []
    for batch_idx, data in enumerate(dataloader):
        if num_batches is not None and batch_idx > num_batches:
            break
        data_cuda = data.to(device)
        optimizer.zero_grad()
        loss = model.step(data_cuda)
        if training:
            loss.backward()
            optimizer.step()
        losses.append(loss.item())
    mean_loss = sum(losses) / len(losses)
    return mean_loss


if __name__ == '__main__':
    # Grid search
    dataset_name = 'Test'
    debug = True

    # num_mel_bins_l = [64, 128, 256]
    # n_fft_l = [512, 1024, 2048]
    # mel_lower_edge_hertz_l = [500]
    # mel_upper_edge_hertz_l = [8000, 12000, 20000]
    # hop_length_ms_l = [5, 10, None]
    # power_l = [1.5]

    num_mel_bins_l = [64]
    n_fft_l = [1024]
    mel_lower_edge_hertz_l = [500]
    mel_upper_edge_hertz_l = [8000]
    hop_length_ms_l = [None]
    win_length_ms_l = [None]
    power_l = [1.5]

    for num_mel_bins, n_fft, mel_lower_edge_hertz, mel_upper_edge_hertz, hop_length_ms, win_length_ms, power in \
            itertools.product(num_mel_bins_l, n_fft_l, mel_lower_edge_hertz_l, mel_upper_edge_hertz_l,
                              hop_length_ms_l, win_length_ms_l, power_l):
        single_file_test(
            num_mel_bins=num_mel_bins,
            n_fft=n_fft,
            mel_lower_edge_hertz=mel_lower_edge_hertz,
            mel_upper_edge_hertz=mel_upper_edge_hertz,
            hop_length_ms=hop_length_ms,
            win_length_ms_l=win_length_ms,
            power=power
        )
