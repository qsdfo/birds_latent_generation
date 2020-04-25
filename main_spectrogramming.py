import pickle

import click
import librosa
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

import matplotlib.pyplot as plt


@click.command()
@click.option('-d', '--debug', is_flag=True)
def main(debug):
    # DATASET_ID = 'BIRD_DB_CATH_segmented'
    DATASET_ID = 'Test_segmented'

    if debug:
        dump_folder = DATA_DIR / 'dump' / 'spectrogramming'

    ################################################################################
    print('Create dataset')
    hparams = HParams(
        sr=44100,
        num_mel_bins=128,
        n_fft=2048,
        mel_lower_edge_hertz=500,
        mel_upper_edge_hertz=8000,
        butter_lowcut=500,
        butter_highcut=8000,
        ref_level_db=20,
        min_level_db=-60,
        mask_spec=False,
        win_length_ms=None,
        hop_length_ms=10,
        mask_spec_kwargs={"spec_thresh": 0.9, "offset": 1e-10},
        n_jobs=1,
        verbosity=1,
        reduce_noise=True,
        noise_reduce_kwargs={"n_std_thresh": 2.0, "prop_decrease": 0.8}
    )
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
                dict_features_to_retain=[],
                key=key,
            )
            for key in tqdm(dataset.data_files.keys())
        )
    syllable_df = pd.concat(syllable_dfs)

    ################################################################################
    print('Get audio for dataset')
    syllable_dfs = []
    for key in syllable_df.key.unique():
        # load audio (key.unique is for loading large wavfiles only once)
        this_syllable_df = syllable_df[syllable_df.key == key]
        wav_loc = dataset.data_files[key].data['wav_loc']
        data = prepare_wav(wav_loc, hparams, dump_folder, debug)
        data = data.astype('float32')
        # get audio for each syllable
        this_syllable_df["audio"] = [
            data[int(st * hparams.sr): int(et * hparams.sr)]
            for st, et in zip(syllable_df.start_time.values, syllable_df.end_time.values)
        ]
        syllable_dfs.append(this_syllable_df)
    syllable_df = pd.concat(syllable_dfs)
    df_mask = np.array([len(i) > 0 for i in tqdm(syllable_df.audio.values)])
    syllable_df = syllable_df[np.array(df_mask)]
    print(f'Number of syllable: {len(syllable_df)}')

    if debug:
        for ind in range(len(syllable_df)):
            if ind > 50:
                break
            librosa.output.write_wav(f'{dump_folder}/syllable_{ind}.wav', syllable_df.audio[ind],
                                     sr=hparams.sr, norm=True)

    # Normalise ???
    ret = []
    for i in syllable_df.audio.values:
        ret.append(i / np.max(i))
    syllable_df['audio'] = ret

    ################################################################################
    print('Create Spectrograms')
    syllables_wav = syllable_df.audio.values
    syllables_spec = []
    for ind, syll_wav in enumerate(syllables_wav):
        # convert to float
        if type(syll_wav[0]) == int:
            syll_wav = int16_to_float32(syll_wav)
        # create spec
        mel_basis = build_mel_basis(hparams, hparams.sr, hparams.sr)
        mel_inversion_basis = build_mel_inversion_basis(mel_basis)
        melspec = spectrogram_librosa(syll_wav, hparams, _mel_basis=mel_basis)
        syllables_spec.append(melspec)

        if debug and (ind < 50):
            plt.clf()
            plt.matshow(melspec, origin="lower")
            plt.savefig(f'{dump_folder}/{ind}_mel_spec.pdf')
            plt.close()

            audio_reconstruct = inv_spectrogram_librosa(melspec, hparams.sr, hparams, mel_inversion_basis=mel_inversion_basis)
            librosa.output.write_wav(f'{dump_folder}/{ind}_reconstruct.wav', audio_reconstruct,
                                     sr=hparams.sr, norm=True)

    ################################################################################
    # print('Rescale Spectrograms')
    # log_scaling_factor = 12
    # with Parallel(n_jobs=n_jobs, verbose=verbosity) as parallel:
    #     syllables_spec = parallel(
    #         delayed(log_resize_spec)(spec, scaling_factor=log_scaling_factor)
    #         for spec in tqdm(syllables_spec, desc="scaling spectrograms", leave=False)
    #     )
    # # Filter syllables badly shaped (often too shorts)
    # syllables_spec = [e for e in syllables_spec if e is not None]
    # if plot:
    #     draw_spec_set(syllables_spec, zoom=1, maxrows=10, colsize=25)

    ################################################################################
    print('Pad Spectrograms')
    # Take 1 secondes max, but a bit more to have square spectrograms, so 128
    # pad_length = int(1 * (1000 / hparams.hop_length_ms))
    pad_length = 128
    syllables_spec_padded = []
    for ind, spec in enumerate(syllables_spec):
        if spec.shape[1] > pad_length:
            spec_padded = None
        else:
            spec_padded = pad_spectrogram(spec, pad_length)
        syllables_spec_padded.append(spec_padded)

        # debug
        if debug and (ind < 50) and (spec_padded is not None):
            mel_basis = build_mel_basis(hparams, hparams.sr, hparams.sr)
            mel_inversion_basis = build_mel_inversion_basis(mel_basis)
            plt.clf()
            plt.matshow(spec_padded, origin="lower")
            plt.savefig(f'{dump_folder}/{ind}_spec_padded.pdf')
            plt.close()
            audio_reconstruct = inv_spectrogram_librosa(spec_padded, hparams.sr, hparams,
                                                        mel_inversion_basis=mel_inversion_basis)
            librosa.output.write_wav(f'{dump_folder}/{ind}_reconstruct_padded.wav', audio_reconstruct,
                                     sr=hparams.sr, norm=True)

    # Save as uint to save space
    syllables_spec_uint = []
    for e in syllables_spec_padded:
        if e is None:
            val = None
        else:
            val = (e * 255).astype('uint8')
        syllables_spec_uint.append(val)

    syllable_df['spectrogram'] = syllables_spec_uint
    syllable_df = syllable_df[syllable_df['spectrogram'].notnull()]
    print(f'Number of syllables after padding: {len(syllable_df)}')

    ################################################################################
    print('Save dataset')
    save_loc = DATA_DIR / 'syllable_dfs' / DATASET_ID / 'data.pickle'
    ensure_dir(save_loc)
    syllable_df.to_pickle(save_loc)
    save_loc = DATA_DIR / 'syllable_dfs' / DATASET_ID / 'hparams.pickle'
    with open(save_loc, 'wb') as ff:
        pickle.dump(hparams, ff)


if __name__ == '__main__':
    main()
