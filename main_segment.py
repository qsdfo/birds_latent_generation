import json
import warnings
from datetime import datetime

import click
import joblib
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from vocalseg.dynamic_thresholding import dynamic_threshold_segmentation
from vocalseg.dynamic_thresholding import plot_segmentations

from avgn.dataset import DataSet
from avgn.signalprocessing.filtering import butter_bandpass_filter
from avgn.utils.audio import load_wav
from avgn.utils.hparams import HParams
from avgn.utils.json import NoIndent, NoIndentEncoder
from avgn.utils.paths import DATA_DIR, ensure_dir


@click.command()
@click.option('-n', '--n_jobs', type=int, default=1)
def main(n_jobs):
    DATASET_ID = 'bird-db'
    # create a unique datetime identifier
    DT_ID = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    # hparams = HParams(
    #     n_fft=4096,
    #     mel_lower_edge_hertz=500,
    #     mel_upper_edge_hertz=20000,
    #     butter_lowcut=500,
    #     butter_highcut=20000,
    #     ref_level_db=20,
    #     min_level_db=-100,
    #     win_length_ms=4,
    #     hop_length_ms=1,
    #     n_jobs=-1,
    #     verbosity=1,
    #     nex=-1
    # )
    hparams = HParams(
        win_length_ms=None,
        hop_length_ms=10,
        n_fft=2048,
        ref_level_db=20,
        min_level_db=-60,
        num_mels_bins=128,
        mel_lower_edge_hertz=500,
        mel_upper_edge_hertz=8000,
        n_jobs=-1,
        verbosity=1,
        nex=-1
    )

    # create a dataset object
    dataset = DataSet(DATASET_ID, hparams=hparams)

    ### segmentation parameters
    n_fft = 4096
    hop_length_ms = 1
    win_length_ms = 4
    ref_level_db = 20
    pre = 0.97
    min_level_db = -100
    min_level_db_floor = -60
    db_delta = 5
    silence_threshold = 0.01
    min_silence_for_spec = 0.05
    max_vocal_for_spec = 1.0,
    min_syllable_length_s = 0.01
    butter_min = 500
    butter_max = 20000
    spectral_range = [500, 20000]

    warnings.filterwarnings("ignore", message="'tqdm_notebook' object has no attribute 'sp'")

    def segment_spec_custom(key, df, save=False, plot=False):
        # load wav
        rate, data = load_wav(df.data["wav_loc"])
        # filter data
        data = butter_bandpass_filter(data, butter_min, butter_max, rate)

        # segment
        results = dynamic_threshold_segmentation(
            data,
            rate,
            n_fft=n_fft,
            hop_length_ms=hop_length_ms,
            win_length_ms=win_length_ms,
            min_level_db_floor=min_level_db_floor,
            db_delta=db_delta,
            ref_level_db=ref_level_db,
            pre=pre,
            min_silence_for_spec=min_silence_for_spec,
            max_vocal_for_spec=max_vocal_for_spec,
            min_level_db=min_level_db,
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
                rate,
            )
            plt.show()

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

        # print(json_txt)

        return results

    indvs = np.array(['_'.join(list(i)) for i in dataset.json_indv])
    nex = -1
    for indv in tqdm(np.unique(indvs), desc="individuals"):
        print(indv)
        indv_keys = np.array(list(dataset.data_files.keys()))[indvs == indv][:nex]

        joblib.Parallel(n_jobs=n_jobs, verbose=11)(
            joblib.delayed(segment_spec_custom)(key, dataset.data_files[key], save=True)
            for key in tqdm(indv_keys, desc="files", leave=False)
        )

if __name__ == '__main__':
    main()