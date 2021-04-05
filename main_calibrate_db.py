"""
Used to test the preprocessing stack.
Not necessary for training or generating
"""

from avgn.signalprocessing.spectrogramming_scipy import build_mel_basis, spectrogram_sp
import os
import shutil
import glob
from avgn.signalprocessing.create_spectrogram_dataset import prepare_wav
from avgn.utils.hparams import HParams
from avgn.utils.paths import DATA_DIR


def calibrate_db(sr, num_mel_bins, n_fft, mel_lower_edge_hertz, mel_upper_edge_hertz, hop_length_ms, win_length_ms,
                 power, ref_level_db, dataset_loc):

    wavs = glob.glob(f'{dataset_loc}/*/*/*.wav')
    min_db = 100000
    max_db = -100000
    for wav_loc in wavs:
        print(wav_loc)
        ret = calibrate_db_file(sr, num_mel_bins, n_fft, mel_lower_edge_hertz, mel_upper_edge_hertz,
                                hop_length_ms, win_length_ms, power, ref_level_db, wav_loc)
        if ret is None:
            continue
        min_db = min(min_db, ret['min_db'])
        max_db = max(max_db, ret['max_db'])
        print(f'min_db: {ret["min_db"]}')
        print(f'max_db: {ret["max_db"]}')
    print(f'min_db: {min_db}')
    print(f'max_db: {max_db}')
    return


def calibrate_db_file(sr, num_mel_bins, n_fft, mel_lower_edge_hertz, mel_upper_edge_hertz, hop_length_ms, win_length_ms,
                      power, ref_level_db, wav_loc):

    hparams = HParams(
        sr=sr,
        num_mel_bins=num_mel_bins,
        n_fft=n_fft,
        mel_lower_edge_hertz=mel_lower_edge_hertz,
        mel_upper_edge_hertz=mel_upper_edge_hertz,
        power=power,  # for spectral inversion
        butter_lowcut=mel_lower_edge_hertz,
        butter_highcut=mel_upper_edge_hertz,
        ref_level_db=ref_level_db,  # 20
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

    dump_folder = 'dump' / f'{suffix}'
    if os.path.isdir(dump_folder):
        shutil.rmtree(dump_folder)
    os.makedirs(dump_folder)

    #  Read wave
    data, _ = prepare_wav(wav_loc, hparams, debug=True)

    # create spec
    if num_mel_bins is not None:
        mel_basis = build_mel_basis(hparams, hparams.sr, hparams.sr)
    else:
        mel_basis = None
    # melspec, debug_info = spectrogram_librosa(data, hparams, _mel_basis=mel_basis, debug=True)
    _, debug_info = spectrogram_sp(y=data,
                                   sr=hparams.sr,
                                   n_fft=hparams.n_fft,
                                   win_length_ms=hparams.win_length_ms,
                                   hop_length_ms=hparams.hop_length_ms,
                                   ref_level_db=hparams.ref_level_db,
                                   _mel_basis=mel_basis,
                                   pre_emphasis=hparams.preemphasis,
                                   power=hparams.power,
                                   debug=True)

    return {
        'max_db': debug_info['max_db'],
        'min_db': debug_info['min_db']
    } if debug_info is not None else None


if __name__ == '__main__':
    # Grid search
    sr = 44100
    num_mel_bins = 64
    n_fft = 1024
    mel_lower_edge_hertz = 1000
    mel_upper_edge_hertz = 12000
    hop_length_ms = None
    win_length_ms = None
    power = 1.5
    ref_level_db = 0
    dataset_loc = '/home/leo/Code/birds_latent_generation/data/raw/voizo_chunks'

    calibrate_db(
        sr=sr,
        num_mel_bins=num_mel_bins,
        n_fft=n_fft,
        mel_lower_edge_hertz=mel_lower_edge_hertz,
        mel_upper_edge_hertz=mel_upper_edge_hertz,
        hop_length_ms=hop_length_ms,
        win_length_ms=win_length_ms,
        power=power,
        ref_level_db=ref_level_db,
        dataset_loc=dataset_loc
    )
