import librosa
import librosa.filters
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal


def spectrogram_librosa(y, hparams, _mel_basis=None, plot=False):
    hop_length = None if hparams.hop_length_ms is None else int(hparams.hop_length_ms / 1000 * hparams.sr)
    win_length = None if hparams.win_length_ms is None else int(hparams.win_length_ms / 1000 * hparams.sr)
    # preprocessing cleaning
    preemphasis_y = preemphasis(y, hparams)  # low-pass filtering to remove noise
    # amplitude spectrum
    S = librosa.stft(y=preemphasis_y, n_fft=hparams.n_fft, hop_length=hop_length, win_length=win_length)
    A = np.abs(S)
    # mel-scale ?
    if _mel_basis is not None:
        A = _linear_to_mel(A**hparams.power, _mel_basis)
        # Mel filters are not normalised to sum to 1 for an output bin of the melspectrogram with librosa...
        # Among other, advantage with normalising is that we can still use decibels afterwards
        # A = melspectrogram(preemphasis_y, sr=fs, n_fft=hparams.n_fft, hop_length=hop_length, win_length=win_length,
        #                      power=1.0, n_mels=hparams.num_mels, fmin=hparams.mel_lower_edge_hertz,
        #                      fmax=hparams.mel_upper_edge_hertz,
        #                      )
    # decibel
    Sdb = librosa.amplitude_to_db(A) - hparams.ref_level_db
    # normalise to [0,1]
    Sdb_norm = _normalize(Sdb, hparams)
    if plot:
        # Preemphasied signal
        librosa.output.write_wav('spectrogramming_test/preemphasis_y.wav', preemphasis_y, sr=22050, norm=True)
        # Normalised spectrogram of preemphasised signal
        fig = plt.figure(dpi=200, figsize=(10, 2))
        im = plt.imshow(Sdb_norm, origin="lower", aspect="auto")
        fig.colorbar(im)
        plt.title('Sdb_norm')
        plt.show()
    return Sdb_norm


def griffinlim_librosa(spectrogram, fs, hparams):
    hop_length = None if hparams.hop_length_ms is None else int(hparams.hop_length_ms / 1000 * fs)
    win_length = None if hparams.win_length_ms is None else int(hparams.win_length_ms / 1000 * fs)
    # We probably don't want to invert the preemphasis in our case
    return librosa.griffinlim(
        spectrogram,
        n_iter=hparams.griffin_lim_iters,
        hop_length=hop_length,
        win_length=win_length,
    )
    # return inv_preemphasis(
    #     librosa.griffinlim(
    #         spectrogram,
    #         n_iter=hparams.griffin_lim_iters,
    #         hop_length=hop_length,
    #         win_length=win_length,
    #     ),
    #     hparams,
    # )


def inv_spectrogram_librosa(spectrogram, fs, hparams, mel_inversion_basis=None):
    """Converts spectrogram to waveform using librosa"""
    if mel_inversion_basis is not None:
        spectrogram = _mel_to_linear(spectrogram, _mel_inverse_basis=mel_inversion_basis)
    S_denorm = _denormalize(spectrogram, hparams)
    S = librosa.db_to_amplitude(
        S_denorm + hparams.ref_level_db
    )  # Convert back to linear
    # Reconstruct phase
    return griffinlim_librosa(S, fs, hparams)


def preemphasis(x, hparams):
    return signal.lfilter([1, -hparams.preemphasis], [1], x)


def inv_preemphasis(x, hparams):
    return signal.lfilter([1], [1, -hparams.preemphasis], x)


def _linear_to_mel(spectrogram, _mel_basis):
    return np.dot(_mel_basis, spectrogram)


def _mel_to_linear(melspectrogram, _mel_basis=None, _mel_inverse_basis=None):
    if (_mel_basis is None) and (_mel_inverse_basis is None):
        raise ValueError("_mel_basis or _mel_inverse_basis needed")
    elif _mel_inverse_basis is None:
        with np.errstate(divide="ignore", invalid="ignore"):
            _mel_inverse_basis = np.nan_to_num(
                np.divide(_mel_basis, np.sum(_mel_basis.T, axis=1))
            ).T
    return np.matmul(_mel_inverse_basis, melspectrogram)


def build_mel_inversion_basis(_mel_basis):
    with np.errstate(divide="ignore", invalid="ignore"):
        mel_inverse_basis = np.nan_to_num(
            np.divide(_mel_basis, np.sum(_mel_basis.T, axis=1))
        ).T
    return mel_inverse_basis


def build_mel_basis(hparams, fs, rate=None, use_n_fft=True):
    if "n_fft" not in hparams.__dict__ or (use_n_fft == False):
        if "num_freq" in hparams.__dict__:
            n_fft = (hparams.num_freq - 1) * 2
        else:
            n_fft = int(hparams.win_length_ms / 1000 * fs)
    else:
        n_fft = hparams.n_fft
    if rate is None:
        rate = hparams.sample_rate
    _mel_basis = librosa.filters.mel(
        rate,
        n_fft,
        n_mels=hparams.num_mel_bins,
        fmin=hparams.mel_lower_edge_hertz,
        fmax=hparams.mel_upper_edge_hertz,
    )
    # Normalise contribution of mel coeff to 1 for 1 output bin
    return np.nan_to_num(_mel_basis.T / np.sum(_mel_basis, axis=1)).T


def _normalize(S, hparams):
    return np.clip((S - hparams.min_level_db) / -hparams.min_level_db, 0, 1)


def _denormalize(S, hparams):
    return (np.clip(S, 0, 1) * -hparams.min_level_db) + hparams.min_level_db
