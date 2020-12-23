import librosa
from matplotlib import pyplot as plt
import soundfile as sf
import random
import re
import shutil
import numpy as np
from pathlib2 import Path
import math
import os


def shorten_wavs(data_dir, dataset_id):

    def new_name(wav_path, chunk_counter):
        new_name = re.sub('.wav', f'_{chunk_counter}.wav', str(wav_path))
        new_name = re.sub(dataset_id, f'{dataset_id}_chunks', new_name)
        return new_name

    dataset_path = Path(f'{data_dir}/raw/{dataset_id}')
    wavs = set((dataset_path).expanduser().glob('**/*.wav'))

    # recreate the dir structure in _chunk dataset
    for x in os.walk(dataset_path):
        subfolder = x[0]
        new_subfolder = re.sub(dataset_id, f'{dataset_id}_chunks', str(subfolder))
        if os.path.isdir(new_subfolder):
            shutil.rmtree(new_subfolder)
        os.mkdir(new_subfolder)

    sr = 44100
    frame_length = 2048
    hop_length = 512
    # we want approx 5 minutes length chunks
    chunk_duration_second = 5 * 60
    min_num_sample = int(chunk_duration_second * sr / hop_length) * hop_length
    for wav_path in wavs:
        # get rate and date
        y, _ = librosa.load(wav_path, sr=sr)
        chunk_counter = 0
        if len(y) > min_num_sample:
            rms = librosa.feature.rms(y, frame_length, hop_length, center=True, pad_mode='reflect')
            rms_zeros = np.where(rms == 0)[1]
            start_t = 0
            for end_win in rms_zeros:
                end_t = end_win * hop_length
                if end_t - start_t < min_num_sample:
                    continue
                else:
                    chunk = y[start_t:end_t]
                    sf.write(new_name(wav_path, chunk_counter), chunk, samplerate=sr)
                    start_t = end_t
                    chunk_counter += 1
            # Don't forget the last chunk
            chunk = y[start_t:]
            sf.write(new_name(wav_path, chunk_counter), chunk, samplerate=sr)
        else:
            sf.write(new_name(wav_path, chunk_counter), y, samplerate=sr)

def fixing_wav_names():
    DATASET_ID = 'voizo'
    dataset_path = Path(f'{DATA_DIR}/raw/{DATASET_ID}')
    wavs = set((dataset_path).expanduser().glob('**/*.wav'))
    for wav in wavs:
        new_wav = re.sub('.aif', '', str(wav))
        shutil.move(wav, new_wav)


def testing_phi0(f0, harm, ampl, sr, duration):
    length = sr * duration
    time = np.arange(0, length)

    period_f0 = math.ceil(sr / f0)

    phi_t_0 = 2 * np.pi * time * f0 / sr

    for _ in range(20):
        f_harm = f0 * harm
        phi0 = random.uniform(0, 2 * np.pi)
        phi_t_harm = 2 * np.pi * time * f_harm / sr + phi0
        x = (np.sin(phi_t_0) + ampl * np.sin(phi_t_harm)) / 2
        plt.plot(list(x[:period_f0]))
        plt.savefig(f'figures/audio/{harm}_{ampl}_{phi0:.4f}.pdf')
        plt.clf()
        sf.write(f'figures/audio/{harm}_{ampl}_{phi0:.4f}.wav', x, sr)


if __name__ == '__main__':
    from avgn.utils.paths import DATA_DIR
    data_dir = DATA_DIR 
    dataset_id = 'voizo'
    shorten_wavs(data_dir, dataset_id)

    # testing_phi0(f0=100, harm=3, ampl=0.8, sr=44100, duration=3)
