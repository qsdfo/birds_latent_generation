# import json
# import warnings
# from datetime import datetime

# import librosa
# import matplotlib.pyplot as plt
# import numpy as np
# from vocalseg.dynamic_thresholding import dynamic_threshold_segmentation
# from vocalseg.dynamic_thresholding import plot_segmentations

# from avgn.dataset import DataSet
# from avgn.signalprocessing.filtering import butter_bandpass_filter
# from avgn.utils.hparams\
#     import HParams
# from avgn.utils.json_custom import NoIndent, NoIndentEncoder
# from avgn.utils.paths import DATA_DIR, ensure_dir


# def main():
#     """
#     min_level_db, min_level_db_floor et delta_db:
#     dynamic segmentation will scan from min_level_db to min_level_db_floor with increment delta_db
#     to find the optimal db threshold level to capture syllables which are:
#      - longer than min_syllable_length_s
#      - smaller than max_vocal_for_spec
#     Need careful tweaking of these five parameters to find the optimal automatic segmentation...
#     """
#     DATASET_ID = 'voizo_chunks_Nigthingale'
#     # DATASET_ID = 'voizo_chunks_Corvus'
#     # create a unique datetime identifier
#     DT_ID = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

#     hparams = HParams(
#         sr=44100,
#         n_fft=4096,
#         mel_lower_edge_hertz=500,
#         mel_upper_edge_hertz=20000,
#         butter_lowcut=500,
#         butter_highcut=20000,
#         ref_level_db=-35,
#         win_length_ms=10,
#         hop_length_ms=2,
#         n_jobs=1,
#         verbosity=1,
#     )

#     # create a dataset object
#     dataset = DataSet(DATASET_ID, hparams=hparams)

#     # segmentation parameters
#     n_fft = hparams.n_fft
#     hop_length_ms = hparams.hop_length_ms
#     win_length_ms = hparams.win_length_ms
#     ref_level_db = hparams.ref_level_db
#     pre = hparams.preemphasis
#     min_level_db = -200 - hparams.ref_level_db
#     min_level_db_floor = -30
#     db_delta = 5
#     silence_threshold = 0.01
#     min_silence_for_spec = 0.05
#     max_vocal_for_spec = 1.0,
#     min_syllable_length_s = 0.2
#     butter_min = hparams.butter_lowcut
#     butter_max = hparams.butter_highcut
#     spectral_range = [hparams.mel_lower_edge_hertz,
#                       hparams.mel_upper_edge_hertz]

#     processed_files = []
#     segmented_files = []

#     def segment_spec_custom(key, df, save=False, plot=False):

#         processed_files.append(key)

#         # load wav
#         data, _ = librosa.core.load(df.data["wav_loc"], sr=hparams.sr)

#         # filter data
#         data = butter_bandpass_filter(data, butter_min, butter_max, hparams.sr)

#         # segment
#         results = dynamic_threshold_segmentation(
#             data,
#             hparams.sr,
#             n_fft=n_fft,
#             hop_length_ms=hop_length_ms,
#             win_length_ms=win_length_ms,
#             min_level_db_floor=min_level_db_floor,
#             db_delta=db_delta,
#             ref_level_db=ref_level_db,
#             pre=pre,
#             min_silence_for_spec=min_silence_for_spec,
#             max_vocal_for_spec=max_vocal_for_spec,
#             min_level_db=min_level_db,
#             silence_threshold=silence_threshold,
#             verbose=True,
#             min_syllable_length_s=min_syllable_length_s,
#             spectral_range=spectral_range,
#         )
#         if results is None:
#             print('skipping')
#             return
#         if plot:
#             plot_segmentations(
#                 results["spec"],
#                 results["vocal_envelope"],
#                 results["onsets"],
#                 results["offsets"],
#                 hop_length_ms,
#                 hparams.sr,
#             )
#             plt.show()

#         segmented_files.append(key)

#         # save the results
#         json_out = DATA_DIR / "processed" / (DATASET_ID + "_segmented") / DT_ID / "JSON" / (
#             key + ".JSON"
#         )

#         json_dict = df.data.copy()

#         json_dict["indvs"][list(df.data["indvs"].keys())[0]]["syllables"] = {
#             "start_times": NoIndent(list(results["onsets"])),
#             "end_times": NoIndent(list(results["offsets"])),
#         }

#         json_txt = json.dumps(json_dict, cls=NoIndentEncoder, indent=2)
#         # save json
#         if save:
#             ensure_dir(json_out.as_posix())
#             print(json_txt, file=open(json_out.as_posix(), "w"))

#         # print(json_txt)

#         return results

#     indvs = np.array(['_'.join(list(i)) for i in dataset.json_indv])
#     for indv in np.unique(indvs):
#         indv_keys = np.array(list(dataset.data_files.keys()))[indvs == indv]
#         for key in indv_keys:
#             print('##############')
#             print(f'# {len(processed_files)}')
#             print(f'# {indv}: {key}')
#             segment_spec_custom(key, dataset.data_files[key], save=True)

#     print(f'Processed files {len(processed_files)}')
#     print(f'Segmented files {len(segmented_files)}')

# if __name__ == '__main__':
#     main()
#     print('done')

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
    DATASET_ID = 'voizo_chunks_Nigthingale'
    # DATASET_ID = 'voizo_chunks_Corvus'
    # create a unique datetime identifier
    DT_ID = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    hparams = HParams(
        sr=44100,
        n_fft=4096,
        mel_lower_edge_hertz=500,
        mel_upper_edge_hertz=20000,
        butter_lowcut=500,
        butter_highcut=20000,
        ref_level_db=-10,
        win_length_ms=10,
        hop_length_ms=2,
        n_jobs=1,
        verbosity=1,
    )

    # create a dataset object
    dataset = DataSet(DATASET_ID, hparams=hparams)

    # segmentation parameters
    n_fft = hparams.n_fft
    hop_length_ms = hparams.hop_length_ms
    win_length_ms = hparams.win_length_ms
    ref_level_db = hparams.ref_level_db
    pre = hparams.preemphasis
    min_level_db_floor = -30
    db_delta = 5
    silence_threshold = 0.01
    min_silence_for_spec = 0.05
    max_vocal_for_spec = 1.0,
    min_syllable_length_s = 0.05
    butter_min = hparams.butter_lowcut
    butter_max = hparams.butter_highcut
    spectral_range = [hparams.mel_lower_edge_hertz,
                      hparams.mel_upper_edge_hertz]

    processed_files = []
    segmented_files = []

    def segment_spec_custom(key, df, save=False, plot=False):

        processed_files.append(key)

        # load wav
        data, _ = librosa.load(df.data["wav_loc"], sr=hparams.sr)

        # filter data
        data = butter_bandpass_filter(data, butter_min, butter_max, hparams.sr)

        # segment
        results = dynamic_threshold_segmentation(
            data,
            hparams.sr,
            n_fft=n_fft,
            hop_length_ms=hop_length_ms,
            win_length_ms=win_length_ms,
            min_level_db_floor=min_level_db_floor,
            db_delta=db_delta,
            ref_level_db=ref_level_db,
            pre=pre,
            min_silence_for_spec=min_silence_for_spec,
            max_vocal_for_spec=max_vocal_for_spec,
            silence_threshold=silence_threshold,
            verbose=True,
            min_syllable_length_s=min_syllable_length_s,
            spectral_range=spectral_range,
        )
        if results is None:
            print('skipping')
            return
        if plot:
            plot_segmentations(
                results["spec"],
                results["vocal_envelope"],
                results["onsets"],
                results["offsets"],
                hop_length_ms,
                hparams.sr,
            )
            plt.show()

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

        # Debug: print start/end times in a text file
        marker_path = re.sub('.wav', '.txt', df.data["wav_loc"])
        with open(marker_path, 'w') as ff:
            for onset, offset in zip(results["onsets"], results["offsets"]):
                ff.write(f"{onset}\t{offset}\n")

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
