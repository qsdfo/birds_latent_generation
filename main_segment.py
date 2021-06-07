from avgn.utils.seconds_to_samples import ms_to_sample
from avgn.signalprocessing.dynamic_thresholding_scipy import dynamic_threshold_segmentation
import json
import re
from datetime import datetime

import librosa
import matplotlib.pyplot as plt
import numpy as np

from avgn.dataset import DataSet
from avgn.signalprocessing.filtering import butter_bandpass_filter
from avgn.utils.hparams\
    import HParams
from avgn.utils.json_custom import NoIndent, NoIndentEncoder
from avgn.utils.paths import DATA_DIR, ensure_dir


def main():
    """
    min_level_db, min_level_db_floor et delta_db:
    dynamic segmentation will scan from min_level_db to min_level_db_floor with increment delta_db
    to find the optimal db threshold level to capture syllables which are:
     - longer than min_syllable_length_s
     - smaller than max_vocal_for_spec
    Need careful tweaking of these five parameters to find the optimal automatic segmentation...
    """
    # DATASET_ID = 'voizo_chunks_Nigthingale'
    # DATASET_ID = 'voizo_chunks_Corvus'
    DATASET_ID = 'du-ra-mo-ni-ro'
    # DATASET_ID = 'voizo_chunks_test'
    # create a unique datetime identifier
    DT_ID = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    # segmentation parameters
    sr = 44100
    n_fft = 4096

    win_length_ms = 10
    hop_length_ms = 2

    mel_lower_edge_hertz = 500
    mel_upper_edge_hertz = 20000
    butter_lowcut = 500
    butter_highcut = 20000
    pre = 0.97

    ref_level_db = -10
    min_level_db_floor = -30
    db_delta = 5

    silence_threshold = 0.01
    min_silence_for_spec = 0.05
    max_vocal_for_spec = 1.0,
    min_syllable_length_s = 0.05
    spectral_range = [mel_lower_edge_hertz,
                      mel_upper_edge_hertz]

    # create an hparam object
    hparams = HParams(
        sr=sr,
        n_fft=n_fft,
        win_length_samples=ms_to_sample(win_length_ms, sr=sr),
        hop_length_samples=ms_to_sample(hop_length_ms, sr=sr),
        chunk_len_samples=None,
        ref_level_db=ref_level_db,
        preemphasis=0.97,
        num_mel_bins=64,
        power=1.5,
        mel_lower_edge_hertz=mel_lower_edge_hertz,
        mel_upper_edge_hertz=mel_upper_edge_hertz,
        butter_lowcut=butter_lowcut,
        butter_highcut=butter_highcut,
        reduce_noise=False,
        noise_reduce_kwargs={},
        mask_spec=False,
        mask_spec_kwargs={"spec_thresh": 0.9, "offset": 1e-10},
        n_jobs=-1,
        verbosity=1
    )
    # create a dataset object
    dataset = DataSet(DATASET_ID, hparams=hparams)

    processed_files = []
    segmented_files = []

    def segment_spec_custom(key, df, save=False, plot=False):

        processed_files.append(key)

        # load wav
        data, _ = librosa.load(df.data["wav_loc"], sr=sr)

        # filter data
        data = butter_bandpass_filter(data, butter_lowcut, butter_highcut, sr)

        # segment
        results = dynamic_threshold_segmentation(
            vocalization=data,
            rate=sr,
            n_fft=n_fft,
            hop_length=ms_to_sample(hop_length_ms, sr),
            win_length=ms_to_sample(win_length_ms, sr),
            min_level_db_floor=min_level_db_floor,
            db_delta=db_delta,
            ref_level_db=ref_level_db,
            pre=pre,
            min_silence_for_spec=min_silence_for_spec,
            max_vocal_for_spec=max_vocal_for_spec,
            silence_threshold=silence_threshold,
            min_syllable_length_s=min_syllable_length_s,
            spectral_range=spectral_range,
            verbose=True,
        )
        if results is None:
            print('skipping')
            return

        segmented_files.append(key)

        # save the results
        json_out = DATA_DIR / "processed" / (DATASET_ID + "_segmented") / DT_ID / "JSON" / (
            key + ".JSON"
        )

        json_dict = df.data.copy()

        json_dict["indvs"][list(df.data["indvs"].keys())[0]]["syllables"] = {
            "start_times": NoIndent(list(results["onsets"])),
            "end_times": NoIndent(list(results["offsets"])),
        }

        json_txt = json.dumps(json_dict, cls=NoIndentEncoder, indent=2)
        # save json
        if save:
            ensure_dir(json_out.as_posix())
            print(json_txt, file=open(json_out.as_posix(), "w"))

        ##########################################
        ##########################################
        # Debug: print start/end times in a text file
        # marker_path = re.sub('.wav', '.txt', df.data["wav_loc"])
        # with open(marker_path, 'w') as ff:
        #     for onset, offset in zip(results["onsets"], results["offsets"]):
        #         ff.write(f"{onset}\t{offset}\n")
        ##########################################
        ##########################################

        return results

    indvs = np.array(['_'.join(list(i)) for i in dataset.json_indv])
    for indv in np.unique(indvs):
        indv_keys = np.array(list(dataset.data_files.keys()))[indvs == indv]
        for key in indv_keys:
            print('##############')
            print(f'# {len(processed_files)}')
            print(f'# {indv}: {key}')
            segment_spec_custom(key, dataset.data_files[key], save=True)

    print(f'Processed files {len(processed_files)}')
    print(f'Segmented files {len(segmented_files)}')

if __name__ == '__main__':
    main()
    print('done')
