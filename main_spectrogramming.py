from avgn.utils.seconds_to_samples import ms_to_sample
from avgn.utils.paths import DATA_DIR
from avgn.utils.audio import int16_to_float32
import itertools
import os
import pickle as pkl
import shutil
import soundfile as sf
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from tqdm import tqdm
from avgn.dataset import DataSet
from avgn.signalprocessing.create_spectrogram_dataset import create_label_df, prepare_wav


def process_syllable(syl, chunk_len_samples):
    # Skip silences
    syl_len = len(syl)
    if syl_len == 0:
        return None, None, None
    if np.max(syl) == 0:
        return None, None, None
    # If too long skip, else pad
    if syl_len > chunk_len_samples:
        return None, None, None
    # Normalise
    sn = syl / np.max(syl)
    # convert to float
    if type(sn[0]) == int:
        sn = int16_to_float32(sn)
    return sn


def main(dataset_id, sr, chunk_len_max_ms, locut, hicut):
    # # STFT time parameters
    # if win_length_ms is None:
    #     win_length = n_fft
    # else:
    #     win_length = ms_to_sample(win_length_ms, sr)
    # if hop_length_ms is None:
    #     hop_length = win_length // 4
    # else:
    #     hop_length = ms_to_sample(hop_length_ms, sr)

    # ################################################################################
    # if chunk_len['type'] == 'ms':
    #     chunk_len_ms = chunk_len['value']
    #     chunk_len_samples_not_rounded = ms_to_sample(chunk_len_ms, sr)
    #     chunk_len_win = round(
    #         (chunk_len_samples_not_rounded - win_length) / hop_length) + 1
    #     chunk_len_samples = (chunk_len_win - 1) * hop_length + win_length
    # elif chunk_len['type'] == 'samples':
    #     chunk_len_samples_not_rounded = chunk_len['value']
    #     chunk_len_win = round(
    #         (chunk_len_samples_not_rounded - win_length) / hop_length) + 1
    #     chunk_len_samples = (chunk_len_win - 1) * hop_length + win_length
    # elif chunk_len['type'] == 'stft_win':
    #     chunk_len_win = chunk_len['value']
    #     chunk_len_samples = (chunk_len_win - 1) * hop_length + win_length
    # ################################################################################
    # print('Chunk length is automatically set to match STFT windows/hop sizes')
    # print(
    #     f'STFT win length: {win_length} samples, {1000 * win_length / sr} ms')
    # print(f'STFT hop length: {hop_length} samples, {1000* hop_length / sr} ms')
    # print(
    #     f'Chunk length: {chunk_len_samples} samples, {chunk_len_win} win, {chunk_len_samples * 1000 / sr} ms')

    ################################################################################
    chunk_len_samples = ms_to_sample(chunk_len_max_ms, sr)
    print('Create dataset')
    suffix = f'sr-{sr}_' +\
        f'chunklmaxs-{chunk_len_samples}_' +\
        f'locut-{locut}_' +\
        f'hicut-{hicut}'
    dataset = DataSet(dataset_id)
    print(f'Number files: {len(dataset.data_files)}')

    ################################################################################
    print('Create a dataset based upon JSON')
    verbosity = 10
    with Parallel(n_jobs=1, verbose=verbosity) as parallel:
        syllable_dfs = parallel(
            delayed(create_label_df)(
                dataset.data_files[key].data,
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
    counter = 0
    save_loc = DATA_DIR / 'syllables' / f'{dataset_id}_{suffix}'
    if os.path.isdir(save_loc):
        raise Exception('already exists')
    os.makedirs(save_loc)
    skipped_counter = 0
    for key in syllable_df.key.unique():
        # load audio (key.unique is for loading large wavfiles only once)
        this_syllable_df = syllable_df[syllable_df.key == key]
        wav_loc = dataset.data_files[key].data['wav_loc']
        print(f'{wav_loc}')
        data = prepare_wav(wav_loc, sr,
                           locut=locut,
                           hicut=hicut,
                           noise_reduce_kwargs={
                               "n_std_thresh": 2.0, "prop_decrease": 0.8}
                           )
        data = data.astype('float32')
        # process each syllable
        for syll_ind, (st, et) in enumerate(zip(this_syllable_df.start_time.values, this_syllable_df.end_time.values)):
            s = data[int(st * sr): int(et * sr)]
            sn = process_syllable(syl=s, chunk_len_samples=chunk_len_samples)
            if sn is None:
                skipped_counter += 1
                continue
            save_dict = {
                'sn': sn,
                'indv': this_syllable_df.indv[syll_ind],
                'label': this_syllable_df.species[syll_ind]
            }
            fname = save_loc / str(counter)
            with open(fname, 'wb') as ff:
                pkl.dump(save_dict, ff)
            counter += 1
    print(f'Skipped counter: {skipped_counter}')

if __name__ == '__main__':
    # dataset_id = 'BIRD_DB_CATH'
    # dataset_id = 'Bird_all'
    # dataset_id = 'Test'
    # dataset_id = 'voizo_all'
    # dataset_id = 'voizo_all_segmented'
    dataset_id = 'voizo_chunks_test_segmented'

    # sr_l = [44100]
    # num_mel_bins_l = [64]
    # # there is not much low frequencies, probably a relatively small value is better for better temporal resolution
    # # 512, but if sr is 16k, we can use a smaller n_fft like 128
    # n_fft_l = [512]
    # # lowest freq = sr / n_fft does not need to be lower
    # # than mel_lower_eddge + gain temporal resolution, but increases number of frames in th STFT.... tradeoff!!
    # mel_lower_edge_hertz_l = [500]
    # mel_upper_edge_hertz_l = [16000]
    # hop_length_ms_l = [None]
    # win_length_ms_l = [None]
    # power_l = [1.5]
    # ref_level_db_l = [-35]
    # # Chunk length can be set either in ms, samples or stft_win
    # # chunk_len = {
    # #     'type': 'ms',
    # #     'value': 1000
    # # }
    #     preemphasis=0.97,
    #     mask_spec=False,
    #     mask_spec_kwargs={"spec_thresh": 0.9, "offset": 1e-10},
    #     reduce_noise=True,
    #     noise_reduce_kwargs={"n_std_thresh": 2.0, "prop_decrease": 0.8},
    #     n_jobs=1,
    #     verbosity=1,
    # chunk_len = {
    #     'type': 'stft_win',
    #     'value': 256
    # }

    sr = 44100
    chunk_len = 1000
    locut = 500
    hicut = 16000
    main(dataset_id, sr, chunk_len, locut, hicut)
    print('done')
