import click
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from joblib import Parallel, delayed
import pandas as pd

from avgn.utils.paths import DATA_DIR, most_recent_subdirectory, ensure_dir

from avgn.utils.hparams import HParams
from avgn.dataset import DataSet
from avgn.signalprocessing.create_spectrogram_dataset import prepare_wav, create_label_df, get_row_audio

from avgn.visualization.spectrogram import draw_spec_set
from avgn.signalprocessing.create_spectrogram_dataset import make_spec, mask_spec, log_resize_spec, pad_spectrogram

import seaborn as sns


@click.command()
@click.option('-n', '--n_jobs', type=int, default=1)
@click.option('--plot', is_flag=True)
def main(n_jobs,
         plot):

    DATASET_ID = 'BIRD_DB_CATH_segmented'

    ################################################################################
    print('Create dataset')
    hparams = HParams(
        num_mel_bins=32,
        mel_lower_edge_hertz=500,
        mel_upper_edge_hertz=10000,
        butter_lowcut=500,
        butter_highcut=10000,
        ref_level_db=20,
        min_level_db=-25,
        mask_spec=True,
        win_length_ms=10,
        hop_length_ms=2,
        mask_spec_kwargs={"spec_thresh": 0.9, "offset": 1e-10},
        n_jobs=n_jobs,
        verbosity=1,
    )
    dataset = DataSet(DATASET_ID, hparams=hparams)
    if plot:
        print(dataset.sample_json)
    print(f'Number files: {len(dataset.data_files)}')

    ################################################################################
    print('Create a dataset based upon JSON')
    verbosity = 10
    with Parallel(n_jobs=n_jobs, verbose=verbosity) as parallel:
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
    if plot:
        print(syllable_df[:3])
    print(f'Number of syllable: {len(syllable_df)}')

    ################################################################################
    print('Get audio for dataset')
    with Parallel(n_jobs=n_jobs, verbose=verbosity) as parallel:
        syllable_dfs = parallel(
            delayed(get_row_audio)(
                syllable_df[syllable_df.key == key],
                dataset.data_files[key].data['wav_loc'],
                dataset.hparams
            )
            for key in tqdm(syllable_df.key.unique())
        )
    syllable_df = pd.concat(syllable_dfs)
    print(f'Number of syllable: {len(syllable_df)}')

    df_mask = np.array([len(i) > 0 for i in tqdm(syllable_df.audio.values)])
    syllable_df = syllable_df[np.array(df_mask)]
    if plot:
        print(syllable_df[:3])

    sylls = syllable_df.audio.values

    if plot:
        nrows = 5
        ncols = 10
        zoom = 2
        fig, axs = plt.subplots(ncols=ncols, nrows=nrows, figsize=(ncols * zoom, nrows + zoom / 1.5))
        for i, syll in tqdm(enumerate(sylls), total=nrows * ncols):
            ax = axs.flatten()[i]
            ax.plot(syll)
            if i == nrows * ncols - 1:
                break

    syllable_df['audio'] = [i / np.max(i) for i in syllable_df.audio.values]

    ################################################################################
    print('Create Spectrograms')
    syllables_wav = syllable_df.audio.values
    syllables_rate = syllable_df.rate.values
    with Parallel(n_jobs=n_jobs, verbose=verbosity) as parallel:
        # create spectrograms
        syllables_spec = parallel(
            delayed(make_spec)(
                syllable,
                rate,
                hparams=dataset.hparams,
                mel_matrix=dataset.mel_matrix,
                use_mel=True,
                use_tensorflow=False,
            )
            for syllable, rate in tqdm(
                zip(syllables_wav, syllables_rate),
                total=len(syllables_rate),
                desc="getting syllable spectrograms",
                leave=False,
            )
        )

    ################################################################################
    print('Rescale Spectrograms')
    log_scaling_factor = 4
    with Parallel(n_jobs=n_jobs, verbose=verbosity) as parallel:
        syllables_spec = parallel(
            delayed(log_resize_spec)(spec, scaling_factor=log_scaling_factor)
            for spec in tqdm(syllables_spec, desc="scaling spectrograms", leave=False)
        )

    if plot:
        draw_spec_set(syllables_spec, zoom=1, maxrows=10, colsize=25)

    ################################################################################
    print('Pad Spectrograms')
    syll_lens = [np.shape(i)[1] for i in syllables_spec]
    pad_length = np.max(syll_lens)
    if plot:
        for indv in np.unique(syllable_df.indv):
            sns.distplot(np.log(syllable_df[syllable_df.indv == indv]["end_time"]), label=indv)
        plt.legend()

    with Parallel(n_jobs=n_jobs, verbose=verbosity) as parallel:
        syllables_spec = parallel(
            delayed(pad_spectrogram)(spec, pad_length)
            for spec in tqdm(
                syllables_spec, desc="padding spectrograms", leave=False
            )
        )

    if plot:
        draw_spec_set(syllables_spec, zoom=1, maxrows=10, colsize=25)

    print(f'Data shape: {np.shape(syllables_spec)}')

    syllable_df['spectrogram'] = syllables_spec

    if plot:
        print(syllable_df[:3])

    ################################################################################
    syllable_df.indv.unique()
    if plot:
        print('View syllables per indv')
        for indv in np.sort(syllable_df.indv.unique()):
            print(indv, np.sum(syllable_df.indv == indv))
            specs = np.array([i / np.max(i) for i in syllable_df[syllable_df.indv == indv].spectrogram.values])
            specs[specs < 0] = 0
            draw_spec_set(specs, zoom=2, maxrows=16, colsize=25)

    ################################################################################
    print('Save dataset')
    save_loc = DATA_DIR / 'syllable_dfs' / DATASET_ID / 'canary.pickle'
    ensure_dir(save_loc)
    syllable_df.to_pickle(save_loc)


if __name__ == '__main__':
    main()


