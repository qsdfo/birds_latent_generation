"""
Used to test the preprocessing stack.
Not necessary for training or generating
"""

import glob
from librosa.core.spectrum import amplitude_to_db
import numpy as np

from scipy import signal
from avgn.signalprocessing.spectrogramming_scipy import _amplitude_to_db, _db_to_amplitude, build_mel_basis, build_mel_inversion_basis, \
    inv_spectrogram_sp, spectrogram_sp
import itertools
import os

import soundfile as sf
import matplotlib.pyplot as plt

from avgn.signalprocessing.create_spectrogram_dataset import prepare_wav
from avgn.utils.hparams import HParams
from avgn.utils.paths import DATA_DIR


def single_file_test(sr, num_mel_bins, n_fft, mel_lower_edge_hertz, mel_upper_edge_hertz, hop_length_ms, win_length_ms,
                     power, ref_level_db, min_level_db, wav_loc, index=0):

    duration_sec = 10

    hparams = HParams(
        sr=sr,
        num_mel_bins=num_mel_bins,
        n_fft=n_fft,
        mel_lower_edge_hertz=mel_lower_edge_hertz,
        mel_upper_edge_hertz=mel_upper_edge_hertz,
        power=power,  # for spectral inversion
        butter_lowcut=mel_lower_edge_hertz,
        butter_highcut=mel_upper_edge_hertz,
        ref_level_db=ref_level_db,
        min_level_db=min_level_db,
        mask_spec=False,
        win_length_ms=win_length_ms,
        hop_length_ms=hop_length_ms,
        mask_spec_kwargs={"spec_thresh": 0.9, "offset": 1e-10},
        n_jobs=1,
        verbosity=1,
        reduce_noise=True,
        noise_reduce_kwargs={"n_std_thresh": 2.0, "prop_decrease": 0.8},
    )
    suffix = hparams.__repr__()

    dump_folder = DATA_DIR / 'dump' / f'{suffix}'
    if not os.path.isdir(dump_folder):
        os.makedirs(dump_folder)

    #  Read wave
    data, _ = prepare_wav(wav_loc, hparams, debug=True)
    data = data[:hparams.sr * duration_sec]

    # create spec
    mel_basis = build_mel_basis(hparams, hparams.sr, hparams.sr)
    mel_inversion_basis = build_mel_inversion_basis(mel_basis)
    # mel_basis = None
    # mel_inversion_basis = None
    plt.clf()
    plt.matshow(mel_basis)
    plt.savefig(f'{dump_folder}/mel_basis.pdf')
    plt.close()
    plt.clf()
    plt.matshow(mel_inversion_basis)
    plt.savefig(f'{dump_folder}/mel_inversion_basis.pdf')
    plt.close()
    _, debug_info = spectrogram_sp(
        data, hparams, _mel_basis=mel_basis, debug=True)

    if debug_info is None:
        print('chunk too short')
        return

    from avgn.signalprocessing.spectrogramming_scipy import griffinlim_sp, _mel_to_linear

    sf.write(f'{dump_folder}/{index}_syllable.wav',
             data, samplerate=hparams.sr)

    #  preemphasis y
    sf.write(f'{dump_folder}/{index}_preemphasis_y.wav',
             debug_info['preemphasis_y'], samplerate=hparams.sr)

    # S_abs
    plt.clf()
    plt.matshow(debug_info['S_abs'][:, :200], origin="lower")
    plt.savefig(f'{dump_folder}/{index}_S_abs.pdf')
    plt.close()
    S_abs_inv = griffinlim_sp(debug_info['S'], hparams.sr, hparams)
    sf.write(f'{dump_folder}/{index}_S_abs.wav',
             S_abs_inv, samplerate=hparams.sr)

    # mel
    plt.clf()
    plt.matshow(debug_info['mel'][:, :200], origin="lower")
    plt.savefig(f'{dump_folder}/{index}_mel.pdf')
    plt.close()
    if mel_basis is not None:
        mel_inv = griffinlim_sp(
            _mel_to_linear(debug_info['mel'], _mel_inverse_basis=mel_inversion_basis),
            hparams.sr, hparams)
        sf.write(f'{dump_folder}/{index}_mel.wav',
                 mel_inv, samplerate=hparams.sr)

    # mel_db
    plt.clf()
    plt.matshow(debug_info['mel_db'][:, :200], origin="lower")
    plt.savefig(f'{dump_folder}/{index}_mel_db.pdf')
    plt.close()
    aa_ = debug_info['mel_db'] + hparams.ref_level_db
    bb_ = _db_to_amplitude(aa_)
    if mel_basis is not None:
        cc_ = _mel_to_linear(bb_, _mel_inverse_basis=mel_inversion_basis)
    else:
        cc_ = bb_
    mel_db_inv = griffinlim_sp(cc_, hparams.sr, hparams)
    sf.write(f'{dump_folder}/{index}_mel_db.wav',
             mel_db_inv, samplerate=hparams.sr)

    ##################################
    ##################################
    f, t, S = signal.stft(x=mel_db_inv, fs=hparams.sr, window='hann', nperseg=1024, noverlap=256,
                          nfft=None, detrend=False, return_onesided=True, boundary='zeros', padded=True, axis=-1)
    S_abs = np.abs(S)
    S_view = _amplitude_to_db(S_abs)
    plt.clf()
    plt.matshow(S_view)
    plt.savefig(f'{dump_folder}/{index}_mel_db_RECON.pdf')
    plt.close()
    ##################################
    ##################################

    # mel_db_norm
    plt.clf()
    plt.matshow(debug_info['mel_db_norm'][:, :200], origin="lower")
    plt.savefig(f'{dump_folder}/{index}_mel_db_norm.pdf')
    plt.close()
    mel_db_norm_inv = inv_spectrogram_sp(debug_info['mel_db_norm'], hparams.sr, hparams,
                                         mel_inversion_basis=mel_inversion_basis)
    sf.write(f'{dump_folder}/{index}_mel_db_norm.wav',
             mel_db_norm_inv, samplerate=hparams.sr)


def db_test(sr, num_mel_bins, n_fft, mel_lower_edge_hertz, mel_upper_edge_hertz, hop_length_ms, win_length_ms,
            power, ref_level_db, min_level_db, dataset_loc):
    wavs = glob.glob(f'{dataset_loc}/*/*/*.wav')
    for index, wav_loc in enumerate(wavs):
        if index >= 3:
            break
        print(wav_loc)
        single_file_test(sr, num_mel_bins, n_fft, mel_lower_edge_hertz, mel_upper_edge_hertz,
                         hop_length_ms, win_length_ms, power, ref_level_db, min_level_db,
                         wav_loc, index)


if __name__ == '__main__':
    # Grid search
    debug = True
    sr_l = [44100]
    num_mel_bins_l = [128]
    n_fft_l = [1024]
    mel_lower_edge_hertz_l = [500]
    mel_upper_edge_hertz_l = [12000]
    hop_length_ms_l = [None]
    win_length_ms_l = [None]
    power_l = [1.5]

    for sr, num_mel_bins, n_fft, mel_lower_edge_hertz, mel_upper_edge_hertz, hop_length_ms, win_length_ms, power in \
            itertools.product(sr_l, num_mel_bins_l, n_fft_l, mel_lower_edge_hertz_l, mel_upper_edge_hertz_l,
                              hop_length_ms_l, win_length_ms_l, power_l):

        dataset_loc = '/home/leo/Code/birds_latent_generation/data/raw/voizo_chunks'

        db_test(
            sr=sr,
            num_mel_bins=num_mel_bins,
            n_fft=n_fft,
            mel_lower_edge_hertz=mel_lower_edge_hertz,
            mel_upper_edge_hertz=mel_upper_edge_hertz,
            hop_length_ms=hop_length_ms,
            win_length_ms=win_length_ms,
            power=power,
            ref_level_db=-20,  # -20
            min_level_db=-180,  # -200 - ref_lvl
            dataset_loc=dataset_loc
        )

        # wav_loc = f'{dataset_loc}/Corvus/XCcorvus-Denoised/Kraai_BieslNp_120312-07xc_0_0.wav'
        # single_file_test(
        #     sr=sr,
        #     num_mel_bins=num_mel_bins,
        #     n_fft=n_fft,
        #     mel_lower_edge_hertz=mel_lower_edge_hertz,
        #     mel_upper_edge_hertz=mel_upper_edge_hertz,
        #     hop_length_ms=hop_length_ms,
        #     win_length_ms=win_length_ms,
        #     power=power,
        #     ref_level_db=-25,
        #     min_level_db=-75,  # -100 - ref_lvl
        #     wav_loc=wav_loc
        # )
