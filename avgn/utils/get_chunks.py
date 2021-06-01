from avgn.pytorch.dataset.spectro_dataset import SpectroDataset
from main_spectrogramming import process_syllable
import os
from avgn.utils.seconds_to_samples import ms_to_sample
from avgn.signalprocessing.create_spectrogram_dataset import prepare_wav
from avgn.signalprocessing.spectrogramming_scipy import build_mel_basis
from avgn.signalprocessing.dynamic_thresholding_scipy import dynamic_threshold_segmentation
from avgn.utils.export_timestamps import export_timestamps


def get_chunks(path, hparams):
    # mel basis
    mel_basis = build_mel_basis(hparams, hparams.sr, hparams.sr)
    # load file
    x_s, _ = prepare_wav(wav_loc=path, hparams=hparams, debug=False)
    # Segmentation params
    min_level_db_floor = -30
    db_delta = 10
    silence_threshold = 0.01
    min_silence_for_spec = 0.05
    max_vocal_for_spec = 1.0
    min_syllable_length_s = 0.05
    # defaults
    # min_level_db_floor = -30
    # db_delta = 5
    # silence_threshold = 0.01
    # min_silence_for_spec = 0.05
    # max_vocal_for_spec = 1.0,
    # min_syllable_length_s = 0.05

    # segment
    results = dynamic_threshold_segmentation(
        x_s,
        hparams.sr,
        n_fft=hparams.n_fft,
        win_length=ms_to_sample(10, sr=hparams.sr),
        hop_length=ms_to_sample(2, sr=hparams.sr),
        min_level_db_floor=min_level_db_floor,
        db_delta=db_delta,
        ref_level_db=hparams.ref_level_db,
        pre=hparams.preemphasis,
        min_silence_for_spec=min_silence_for_spec,
        max_vocal_for_spec=max_vocal_for_spec,
        silence_threshold=silence_threshold,
        verbose=True,
        min_syllable_length_s=min_syllable_length_s,
        spectral_range=[hparams.mel_lower_edge_hertz,
                        hparams.mel_upper_edge_hertz],
    )
    if results is None:
        print('Cannot segment the input file')
        return
    # chunks
    start_times = results["onsets"]
    end_times = results["offsets"]
    chunks_mS = []
    start_samples = []
    end_samples = []
    name_no_ext = os.path.splitext(os.path.split(path)[1])[0]
    export_timestamps(start_times, end_times, f'dump/{name_no_ext}.txt')
    for start_time, end_time in zip(start_times, end_times):
        start_sample = int(start_time * hparams.sr)
        end_sample = int(end_time * hparams.sr)
        syl = x_s[start_sample:end_sample]

        # To avoid mistakes, reproduce the whole preprocessing pipeline, even (here useless) int casting
        _, mS, _ = process_syllable(
            syl, hparams, mel_basis=mel_basis, debug=False)
        if mS is None:
            continue
        mS_int = (mS * 255).astype('uint8')
        sample = SpectroDataset.process_mSp(mS_int, chunk_len_win=hparams.chunk_len_win,
                                            num_mel_bins=hparams.num_mel_bins)

        chunks_mS.append(sample)
        start_samples.append(start_sample)
        end_samples.append(end_sample)
    return x_s, chunks_mS, start_samples, end_samples


def get_contam_chunks(path, source_start_samples, source_end_samples, hparams):
    # mel basis
    mel_basis = build_mel_basis(hparams, hparams.sr, hparams.sr)
    # load file
    x_s, _ = prepare_wav(wav_loc=path, hparams=hparams, debug=False)
    # chunks
    start_times = [i / hparams.sr for i in source_start_samples]
    end_times = [i / hparams.sr for i in source_end_samples]
    chunks_mS = []
    start_samples = []
    end_samples = []
    # export_timestamps(start_times, end_times, 'dump/target_timestamps.txt')
    for start_time, end_time in zip(start_times, end_times):
        start_sample = int(start_time * hparams.sr)
        end_sample = int(end_time * hparams.sr)
        syl = x_s[start_sample:end_sample]

        # To avoid mistakes, reproduce the whole preprocessing pipeline, even (here useless) int casting
        _, mS, _ = process_syllable(
            syl, hparams, mel_basis=mel_basis, debug=False)
        if mS is None:
            continue
        mS_int = (mS * 255).astype('uint8')
        sample = SpectroDataset.process_mSp(mS_int)

        chunks_mS.append(sample)
        start_samples.append(start_sample)
        end_samples.append(end_sample)
    return x_s, chunks_mS, start_samples, end_samples
