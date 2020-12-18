from matplotlib import pyplot as plt
import soundfile as sf
import random
import re
import shutil
import numpy as np
from avgn.utils.paths import DATA_DIR
from pathlib2 import Path
import math


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
    testing_phi0(f0=100, harm=3, ampl=0.8, sr=44100, duration=3)
