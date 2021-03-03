import librosa
import six
import numpy as np
from scipy import signal


def spectrogram_sp(y, hparams, _mel_basis=None, debug=False):
    win_length = hparams.n_fft if hparams.win_length_ms is None else int(hparams.win_length_ms / 1000 * hparams.sr)
    hop_length = win_length // 4 if hparams.hop_length_ms is None else int(hparams.hop_length_ms / 1000 * hparams.sr)
    overlap_length = win_length - hop_length
    # preprocessing cleaning
    preemphasis_y = preemphasis(y, hparams)  # low-pass filtering to remove noise

    # Check chunk len
    if len(preemphasis_y) < win_length:
        return None, None

    # amplitude spectrum
    f, t, S = signal.stft(x=preemphasis_y, fs=hparams.n_fft, window='hann', nperseg=win_length, noverlap=overlap_length,
                          nfft=hparams.n_fft, detrend=False, return_onesided=True, boundary='zeros', padded=True, axis=- 1)
    S_abs = np.abs(S)
    # mel-scale ?
    if _mel_basis is not None:
        A = _linear_to_mel(S_abs, _mel_basis)
    else:
        A = S_abs
    # decibel
    Sdb_unref = _amplitude_to_db(A)
    max_db = Sdb_unref.max()
    min_db = Sdb_unref.min()
    Sdb = Sdb_unref - hparams.ref_level_db
    # normalise to [0,1]
    Sdb_norm = _normalize(Sdb, hparams)
    if debug:
        debug_info = {
            'preemphasis_y': preemphasis_y,
            'S': S,
            'S_abs': S_abs,
            'mel': A,
            'mel_db': Sdb,
            'mel_db_norm': Sdb_norm,
            'max_db': max_db,
            'min_db': min_db
        }
    else:
        debug_info = None
    return Sdb_norm, debug_info


def griffinlim_sp(spectrogram, fs, hparams):
    win_length = hparams.n_fft if hparams.win_length_ms is None else int(hparams.win_length_ms / 1000 * hparams.sr)
    hop_length = win_length // 4 if hparams.hop_length_ms is None else int(hparams.hop_length_ms / 1000 * hparams.sr)
    overlap_length = win_length - hop_length
    t, s = signal.istft(spectrogram, fs=fs, window='hann', nperseg=win_length, noverlap=overlap_length,
                        nfft=None, input_onesided=True, boundary=True, time_axis=-1, freq_axis=-2)
    return s / np.max(np.abs(s))


def inv_spectrogram_sp(spectrogram, fs, hparams, mel_inversion_basis=None):
    """Converts spectrogram to waveform using librosa"""
    s_unnorm = _denormalize(spectrogram, hparams)
    s_amplitude = _db_to_amplitude(s_unnorm + hparams.ref_level_db)
    if mel_inversion_basis is not None:
        s_linear = _mel_to_linear(s_amplitude, _mel_inverse_basis=mel_inversion_basis)
    else:
        s_linear = s_amplitude
    return griffinlim_sp(s_linear, fs, hparams)


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
    return (np.clip(S, 0, 1) * - hparams.min_level_db) + hparams.min_level_db


def _amplitude_to_db(S, amin=1e-5):
    S = np.asarray(S)
    magnitude = np.abs(S)
    power = np.square(magnitude, out=magnitude)
    return _power_to_db(power, amin=amin**2)


def _power_to_db(S, amin=1e-10):
    S = np.asarray(S)
    if amin <= 0:
        raise ValueError('amin < 0')
    log_spec = 10.0 * np.log10(np.maximum(amin, S))
    return log_spec


def _db_to_power(S_db):
    return np.power(10.0, 0.1 * S_db)


def _db_to_amplitude(S_db):
    return _db_to_power(S_db)**0.5
