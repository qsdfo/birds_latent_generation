import itertools
import os
import pickle as pkl
import shutil
import soundfile as sf
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from tqdm import tqdm

from avgn.dataset import DataSet
from avgn.signalprocessing.create_spectrogram_dataset import create_label_df, prepare_wav, pad_spectrogram
from avgn.signalprocessing.spectrogramming import build_mel_basis, build_mel_inversion_basis, spectrogram_librosa, \
    inv_spectrogram_librosa
from avgn.utils.audio import int16_to_float32
from avgn.utils.hparams import HParams
from avgn.utils.paths import DATA_DIR, ensure_dir


def process_syllable(syl, hparams, mel_basis, debug):
    # Skip silences
    if len(syl) == 0:
        return None, None, None
    if np.max(syl) == 0:
        return None, None, None
    # Normalise
    sn = syl / np.max(syl)
    # convert to float
    if type(sn[0]) == int:
        sn = int16_to_float32(sn)
    # create spec
    mS, debug_info = spectrogram_librosa(
        sn, hparams, _mel_basis=mel_basis, debug=debug)
    if mS.shape[1] > hparams.pad_length:
        # Just skip that syllable if its too long
        return None, None, None
    else:
        mSp = pad_spectrogram(mS, hparams.pad_length)
    return sn, mSp, debug_info


def main(debug, num_mel_bins, n_fft, pad_length, mel_lower_edge_hertz, mel_upper_edge_hertz,
         hop_length_ms, win_length_ms, power):
    # DATASET_ID = 'BIRD_DB_CATH'
    # DATASET_ID = 'Bird_all'
    # DATASET_ID = 'Test'
    DATASET_ID = 'voizo_all'
    ind_examples = [20, 40, 50, 60, 80, 100]

    ################################################################################
    print('Create dataset')
    hparams = HParams(
        sr=44100,
        num_mel_bins=num_mel_bins,
        n_fft=n_fft,
        pad_length=pad_length,
        mel_lower_edge_hertz=mel_lower_edge_hertz,
        mel_upper_edge_hertz=mel_upper_edge_hertz,
        power=power,  # for spectral inversion
        butter_lowcut=mel_lower_edge_hertz,
        butter_highcut=mel_upper_edge_hertz,
        ref_level_db=40,
        min_level_db=-90,
        mask_spec=False,
        win_length_ms=win_length_ms,
        hop_length_ms=hop_length_ms,
        mask_spec_kwargs={"spec_thresh": 0.9, "offset": 1e-10},
        n_jobs=1,
        verbosity=1,
        reduce_noise=True,
        noise_reduce_kwargs={"n_std_thresh": 2.0, "prop_decrease": 0.8},
        griffin_lim_iters=50
    )
    suffix = hparams.__repr__()

    if debug:
        dump_folder = DATA_DIR / 'dump' / f'{suffix}'
        if os.path.isdir(dump_folder):
            shutil.rmtree(dump_folder)
        os.makedirs(dump_folder)
    else:
        dump_folder = None

    dataset = DataSet(DATASET_ID, hparams=hparams)
    print(f'Number files: {len(dataset.data_files)}')

    ################################################################################
    print('Create a dataset based upon JSON')
    verbosity = 10
    with Parallel(n_jobs=1, verbose=verbosity) as parallel:
        syllable_dfs = parallel(
            delayed(create_label_df)(
                dataset.data_files[key].data,
                hparams=dataset.hparams,
                labels_to_retain=[],
                unit="syllables",
                dict_features_to_retain=['species'],
                key=key,
            )
            for key in tqdm(dataset.data_files.keys())
        )
    syllable_df = pd.concat(syllable_dfs)

    ################################################################################
    print('Get audio for dataset')
    mel_basis = build_mel_basis(hparams, hparams.sr, hparams.sr)
    mel_inversion_basis = build_mel_inversion_basis(mel_basis)
    counter = 0
    save_loc = DATA_DIR / 'syllables' / f'{DATASET_ID}_{suffix}'
    if os.path.isdir(save_loc):
        raise Exception('already exists')
    os.makedirs(save_loc)
    skipped_counter = 0
    for key in syllable_df.key.unique():
        # load audio (key.unique is for loading large wavfiles only once)
        this_syllable_df = syllable_df[syllable_df.key == key]
        wav_loc = dataset.data_files[key].data['wav_loc']
        print(f'{wav_loc}')
        data, _ = prepare_wav(wav_loc, hparams, debug=debug)
        data = data.astype('float32')
        # process each syllable
        for syll_ind, (st, et) in enumerate(zip(this_syllable_df.start_time.values, this_syllable_df.end_time.values)):
            s = data[int(st * hparams.sr): int(et * hparams.sr)]
            sn, mSp, debug_info = process_syllable(
                syl=s, hparams=hparams, mel_basis=mel_basis, debug=debug)
            if mSp is None:
                skipped_counter += 1
                continue
            # Save as uint to save space
            val = (mSp * 255).astype('uint8')
            save_dict = {
                'mSp': val,
                'sn': sn,
                'indv': this_syllable_df.indv[syll_ind],
                'label': this_syllable_df.species[syll_ind]
            }
            fname = save_loc / str(counter)
            with open(fname, 'wb') as ff:
                pkl.dump(save_dict, ff)
            counter += 1

            if debug and (counter in ind_examples):
                # normalised audio
                sf.write(f'{dump_folder}/{counter}_sn.wav',
                         sn, samplerate=hparams.sr)
                #  Padded mel db norm spectro
                plt.clf()
                plt.matshow(mSp, origin="lower")
                plt.savefig(f'{dump_folder}/{counter}_mSp.pdf')
                plt.close()
                audio_reconstruct = inv_spectrogram_librosa(mSp, hparams.sr, hparams,
                                                            mel_inversion_basis=mel_inversion_basis)
                sf.write(f'{dump_folder}/{counter}_mSp.wav',
                         audio_reconstruct, samplerate=hparams.sr)

    print(f'Skipped counter: {skipped_counter}')
    #  Save hparams
    print("Save hparams")
    hparams_loc = f'{save_loc}_hparams.pkl'
    with open(hparams_loc, 'wb') as ff:
        pkl.dump(hparams, ff)


if __name__ == '__main__':
    # Grid search
    debug = True
    num_mel_bins_l = [64]
    n_fft_l = [1024]
    mel_lower_edge_hertz_l = [500]
    mel_upper_edge_hertz_l = [8000]
    hop_length_ms_l = [None]
    win_length_ms_l = [None]
    power_l = [1.5]
    # We want spectro representing num_seconds of signal
    # pad_length = int(num_seconds / hparams.hop_length_seconds))
    # with, if hparams.hop_length_ms = None, hop_length_seconds = (hparams.sr / (hparams.n_fft / 4))
    # so pad_length = num_seconds * sr / (n_fft/4)
    # For 1 second, with sr = 44100, n_fft = 1024, pad_length = 172.260
    pad_length = 128

    for num_mel_bins, n_fft, mel_lower_edge_hertz, mel_upper_edge_hertz, hop_length_ms, win_length_ms, power in \
            itertools.product(num_mel_bins_l, n_fft_l, mel_lower_edge_hertz_l, mel_upper_edge_hertz_l,
                              hop_length_ms_l, win_length_ms_l, power_l):
        main(debug=debug,
             num_mel_bins=num_mel_bins,
             n_fft=n_fft,
             pad_length=pad_length,
             mel_lower_edge_hertz=mel_lower_edge_hertz,
             mel_upper_edge_hertz=mel_upper_edge_hertz,
             hop_length_ms=hop_length_ms,
             win_length_ms=win_length_ms,
             power=power)
