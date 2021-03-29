import time
import librosa
from scipy import signal

def timing_stft(wave_path):

    n_fft = 1024
    sr = 44100
    win_length = n_fft
    hop_length = n_fft // 4
    overlap_length = win_length - hop_length

    # load wav
    y, _ = librosa.load(wave_path, sr=sr)

    # librosa
    start_time = time.time()
    _ = librosa.stft(y=y, n_fft=n_fft, hop_length=hop_length, win_length=win_length)
    end_time = time.time()
    print(f'Librosa time: {end_time - start_time} seconds')

    # scipy
    start_time = time.time()
    _ = signal.stft(x=y, fs=sr, nperseg=win_length, noverlap=overlap_length, nfft=n_fft)
    end_time = time.time()
    print(f'Scipy time: {end_time - start_time} seconds')

if __name__ == '__main__':
    wave_path = "/home/leo/Code/birds_latent_generation/data/raw/voizo_chunks/Corvus/XCcorvus-Denoised/XC467664-190419_02B sel Korp taltrast rödhake skäggdopping_0_0.wav"
    timing_stft(wave_path)
