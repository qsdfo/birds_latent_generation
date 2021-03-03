import re
import os
import numpy as np
from pathlib2 import Path
import shutil
import soundfile as sf
import librosa


def chunk_files(data_dir, dataset_id):

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
    # we want approx 2 minutes length chunks
    chunk_duration_second = 2 * 60
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
        del y
    print('finished')
if __name__ == '__main__':
    from avgn.utils.paths import DATA_DIR
    data_dir = DATA_DIR
    dataset_id = 'voizo'
    chunk_files(data_dir, dataset_id)
