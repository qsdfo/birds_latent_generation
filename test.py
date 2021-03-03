from matplotlib import pyplot as plt
import re
from pathlib2 import Path
from scipy.io import wavfile


def shorten_wavs(data_dir, dataset_id):
    def new_name(wav_path, chunk_counter):
        new_name = re.sub('.wav', f'_{chunk_counter}.wav', str(wav_path))
        new_name = re.sub(dataset_id, f'{dataset_id}_chunks', new_name)
        return new_name
    dataset_path = Path(f'{data_dir}/raw/{dataset_id}')
    wavs = set((dataset_path).expanduser().glob('**/*.wav'))
    print(dataset_path)
    sr = 44100
    frame_length = 2048
    hop_length = 512
    # we want approx 5 minutes length chunks
    chunk_duration_second = 2 * 60
    min_num_sample = int(chunk_duration_second * sr / hop_length) * hop_length
    for wav_path in wavs:
        print(wav_path)
        samplerate, data = wavfile.read(wav_path)
        print(samplerate)

if __name__ == '__main__':
    PROJECT_DIR = Path(__file__).resolve().parents[0]
    data_dir = PROJECT_DIR / "data"
    print(data_dir)
    dataset_id = 'voizo'
    shorten_wavs(data_dir, dataset_id)