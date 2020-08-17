import collections

import librosa
import noisereduce as nr
import numpy as np
import pandas as pd
from PIL import Image
from joblib import Parallel, delayed
from tqdm.autonotebook import tqdm

from avgn.signalprocessing.filtering import butter_bandpass_filter
from avgn.utils.audio import int16_to_float32


def flatten_spectrograms(specs):
    return np.reshape(specs, (np.shape(specs)[0], np.prod(np.shape(specs)[1:])))


def norm(x):
    return (x - np.min(x)) / (np.max(x) - np.min(x))


def log_resize_spec(spec, scaling_factor=10):
    resize_shape = [int(np.log(max(np.shape(spec)[1], 2)) * scaling_factor), np.shape(spec)[0]]
    resize_spec = np.array(Image.fromarray(spec).resize(resize_shape, Image.ANTIALIAS))
    return resize_spec


def pad_spectrogram(spectrogram, pad_length):
    """ Pads a spectrogram to being a certain length
    """
    excess_needed = pad_length - np.shape(spectrogram)[1]
    pad_left = np.floor(float(excess_needed) / 2).astype("int")
    pad_right = np.ceil(float(excess_needed) / 2).astype("int")
    return np.pad(
        spectrogram, [(0, 0), (pad_left, pad_right)], "constant", constant_values=0
    )


def list_match(_list, list_of_lists):
    # Using Counter
    return [
        collections.Counter(elem) == collections.Counter(_list)
        for elem in list_of_lists
    ]


def mask_spec(spec, spec_thresh=0.9, offset=1e-10):
    """ mask threshold a spectrogram to be above some % of the maximum power
    """
    mask = spec >= (spec.max(axis=0, keepdims=1) * spec_thresh + offset)
    return spec * mask


def create_syllable_df(
        dataset,
        indv,
        unit="syllables",
        log_scaling_factor=10,
        verbosity=0,
        log_scale_time=True,
        pad_syllables=True,
        n_jobs=-1,
        include_labels=False,
):
    """ from a DataSet object, get all of the syllables from an individual as a spectrogram
    """
    with tqdm(total=4) as pbar:
        # get waveform of syllables
        pbar.set_description("getting syllables")
        with Parallel(n_jobs=n_jobs, verbose=verbosity) as parallel:
            syllables = parallel(
                delayed(subset_syllables)(
                    json_file,
                    indv=indv,
                    unit=unit,
                    hparams=dataset.hparams,
                    include_labels=include_labels,
                )
                for json_file in tqdm(
                    np.array(dataset.json_files)[list_match(indv, dataset.json_indv)],
                    desc="getting syllable wavs",
                    leave=False,
                )
            )

            # repeat rate for each wav
            syllables_sequence_id = np.concatenate(
                [np.repeat(ii, len(i[0])) for ii, i in enumerate(syllables)]
            )
            syllables_sequence_pos = np.concatenate(
                [np.arange(len(i[0])) for ii, i in enumerate(syllables)]
            )

            # list syllables waveforms
            syllables_wav = [
                item for sublist in [i[0] for i in syllables] for item in sublist
            ]

            # repeat rate for each wav
            syllables_rate = np.concatenate(
                [np.repeat(i[1], len(i[0])) for i in syllables]
            )

            # list syllable labels
            if syllables[0][2] is not None:
                syllables_labels = np.concatenate([i[2] for i in syllables])
            else:
                syllables_labels = None
            pbar.update(1)
            pbar.set_description("creating spectrograms")

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

            # Mask spectrograms
            if dataset.hparams.mask_spec:
                syllables_spec = parallel(
                    delayed(mask_spec)(syllable, **dataset.hparams.mask_spec_kwargs)
                    for syllable in tqdm(
                        syllables_spec,
                        total=len(syllables_rate),
                        desc="masking spectrograms",
                        leave=False,
                    )
                )

            pbar.update(1)
            pbar.set_description("rescaling syllables")
            # log resize spectrograms
            if log_scale_time:
                syllables_spec = parallel(
                    delayed(log_resize_spec)(spec, scaling_factor=log_scaling_factor)
                    for spec in tqdm(
                        syllables_spec, desc="scaling spectrograms", leave=False
                    )
                )
            pbar.update(1)
            pbar.set_description("padding syllables")
            # determine padding
            syll_lens = [np.shape(i)[1] for i in syllables_spec]
            pad_length = np.max(syll_lens)

            # pad syllables
            if pad_syllables:
                syllables_spec = parallel(
                    delayed(pad_spectrogram)(spec, pad_length)
                    for spec in tqdm(
                        syllables_spec, desc="padding spectrograms", leave=False
                    )
                )
            pbar.update(1)

            syllable_df = pd.DataFrame(
                {
                    "syllables_sequence_id": syllables_sequence_id,
                    "syllables_sequence_pos": syllables_sequence_pos,
                    "syllables_wav": syllables_wav,
                    "syllables_rate": syllables_rate,
                    "syllables_labels": syllables_labels,
                    "syllables_spec": syllables_spec,
                }
            )

        return syllable_df


def prepare_wav(wav_loc, hparams, dump_folder, debug):
    """ load wav and convert to correct format
    """

    # get rate and date
    # rate, data = load_wav(wav_loc)

    data, _ = librosa.core.load(wav_loc, sr=hparams.sr)
    if debug:
        librosa.output.write_wav(f'{dump_folder}/original.wav', data, sr=hparams.sr, norm=True)

    # convert data if needed
    if np.issubdtype(type(data[0]), np.integer):
        data = int16_to_float32(data)

    #######################################
    #######################################
    #######################################"
    # # bandpass filter
    # if hparams is not None:
    #     data_TTT = butter_bandpass_filter(
    #         data, hparams.butter_lowcut, hparams.butter_highcut, hparams.sr, order=5
    #     )
    #     if debug:
    #         librosa.output.write_wav(f'{dump_folder}/butter_bandpass_filter.wav', data, sr=hparams.sr, norm=True)
    #
    #     # reduce noise
    #     if hparams.reduce_noise:
    #         data_TTT = nr.reduce_noise(
    #             audio_clip=data_TTT, noise_clip=data_TTT, **hparams.noise_reduce_kwargs
    #         )
    #######################################
    #######################################
    #######################################

    # Chunks to avoid memory issues
    len_chunk_minutes = 30
    len_chunk_sample = hparams.sr * 60 * len_chunk_minutes
    data_chunks = []
    for t in range(0, len(data), len_chunk_sample):
        start = t
        end = min(len(data), t + len_chunk_sample)
        data_chunks.append(data[start:end])

    # bandpass filter
    data_cleaned = []
    if hparams is not None:
        for data in data_chunks:
            data = butter_bandpass_filter(
                data, hparams.butter_lowcut, hparams.butter_highcut, hparams.sr, order=5
            )
            if debug:
                librosa.output.write_wav(f'{dump_folder}/butter_bandpass_filter.wav', data, sr=hparams.sr, norm=True)

            # reduce noise
            if hparams.reduce_noise:
                data = nr.reduce_noise(
                    audio_clip=data, noise_clip=data, **hparams.noise_reduce_kwargs
                )
            data_cleaned.append(data)
    else:
        data_cleaned = data_chunks

    #  concatenate chunks
    data = np.concatenate(data_cleaned)

    if debug:
        librosa.output.write_wav(f'{dump_folder}/reduce_noise.wav', data, sr=hparams.sr, norm=True)

    return data


def create_label_df(
        json_dict,
        hparams=None,
        labels_to_retain=[],
        unit="syllables",
        dict_features_to_retain=[],
        key=None,
):
    """ create a dataframe from json dictionary of time events and labels
    """

    syllable_dfs = []
    # loop through individuals
    for indvi, indv in enumerate(json_dict["indvs"].keys()):
        if unit not in json_dict["indvs"][indv].keys():
            continue
        indv_dict = {}
        indv_dict["start_time"] = json_dict["indvs"][indv][unit]["start_times"]
        indv_dict["end_time"] = json_dict["indvs"][indv][unit]["end_times"]

        # get data for individual
        for label in labels_to_retain:
            indv_dict[label] = json_dict["indvs"][indv][unit][label]
            if len(indv_dict[label]) < len(indv_dict["start_time"]):
                indv_dict[label] = np.repeat(
                    indv_dict[label], len(indv_dict["start_time"])
                )

        # create dataframe
        indv_df = pd.DataFrame(indv_dict)
        indv_df["indv"] = indv
        indv_df["indvi"] = indvi
        syllable_dfs.append(indv_df)

    syllable_df = pd.concat(syllable_dfs)
    for feat in dict_features_to_retain:
        syllable_df[feat] = json_dict[feat]
    # associate current syllables with key
    syllable_df["key"] = key

    return syllable_df


def get_row_audio(syllable_df, wav_loc, hparams):
    """ load audio and grab individual syllables
    TODO: for large sparse WAV files, the audio should be loaded only for the syllable
    """

    # load audio
    rate, data = prepare_wav(wav_loc, hparams)
    data = data.astype('float32')

    # get audio for each syllable
    syllable_df["audio"] = [
        data[int(st * rate):int(et * rate)]
        for st, et in zip(syllable_df.start_time.values, syllable_df.end_time.values)
    ]

    syllable_df["rate"] = rate

    return syllable_df
