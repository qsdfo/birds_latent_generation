import librosa
import numpy as np
from scipy import signal


def spectrogram_sp(y, sr, n_fft, win_length, hop_length, ref_level_db, _mel_basis, pre_emphasis, power, debug):
    if win_length is None:
        win_length = n_fft
    if hop_length is None:
        hop_length = win_length // 4
    overlap_length = win_length - hop_length
    # preprocessing cleaning
    # low-pass filtering to remove noise
    preemphasis_y = preemphasis(y, pre_emphasis)

    # Check chunk len
    if len(preemphasis_y) < win_length:
        return None, None

    # amplitude spectrum
    f, t, S = signal.stft(x=preemphasis_y, fs=sr, window='hann', nperseg=win_length, noverlap=overlap_length,
                          nfft=n_fft, detrend=False, return_onesided=True, boundary=None, padded=True, axis=-1)
    S_abs = np.abs(S)
    # mel-scale ?
    if _mel_basis is not None:
        A = _linear_to_mel(S_abs**power, _mel_basis)
    else:
        A = S_abs
    # decibel
    Sdb_unref = _amplitude_to_db(A)
    max_db = Sdb_unref.max()
    min_db = Sdb_unref.min()
    Sdb_norm = _normalize(
        Sdb_unref, min_db=_min_level_db(), max_db=ref_level_db)
    if debug:
        debug_info = {
            'preemphasis_y': preemphasis_y,
            'S': S,
            'S_abs': S_abs,
            'mel': A,
            'mel_db': Sdb_unref,
            'mel_db_norm': Sdb_norm,
            'max_db': max_db,
            'min_db': min_db
        }
    else:
        debug_info = None
    return Sdb_norm, debug_info


def griffinlim_sp(spectrogram, n_fft, win_length, hop_length):
    if win_length is None:
        win_length = n_fft
    if hop_length is None:
        hop_length = win_length // 4
    s = librosa.griffinlim(
        spectrogram,
        n_iter=50,
        hop_length=hop_length,
        win_length=win_length,
        momentum=0.5,
    )
    return s / np.max(np.abs(s))


def inv_spectrogram_sp(spectrogram, n_fft, win_length, hop_length, ref_level_db, power, mel_inversion_basis):
    """Converts spectrogram to waveform using librosa"""
    s_unnorm = _denormalize(
        spectrogram, min_db=_min_level_db(), max_db=ref_level_db)
    s_amplitude = _db_to_amplitude(s_unnorm + ref_level_db)
    if mel_inversion_basis is not None:
        s_linear = _mel_to_linear(
            s_amplitude, _mel_inverse_basis=mel_inversion_basis)**(1 / power)
        # s_linear = _mel_to_linear(s_amplitude, _mel_inverse_basis=mel_inversion_basis)
    else:
        s_linear = s_amplitude
    return griffinlim_sp(s_linear, n_fft=n_fft, win_length=win_length, hop_length=hop_length)


def preemphasis(x, pre_emphasis):
    return signal.lfilter([1, -pre_emphasis], [1], x)


def inv_preemphasis(x, pre_emphasis):
    return signal.lfilter([1], [1, -pre_emphasis], x)


def _linear_to_mel(spectrogram, _mel_basis):
    return np.dot(_mel_basis, spectrogram)


def _mel_to_linear(melspectrogram, _mel_inverse_basis):
    return np.matmul(_mel_inverse_basis, melspectrogram)


def build_mel_inversion_basis(_mel_basis):
    mel_inverse_basis = np.divide(_mel_basis, np.sum(
        (_mel_basis + _epsilon()).T, axis=1)).T
    return mel_inverse_basis


def build_mel_basis(hparams, fs, rate=None, use_n_fft=True):
    if "n_fft" not in hparams.__dict__ or not use_n_fft:
        if "num_freq" in hparams.__dict__:
            n_fft = (hparams.num_freq - 1) * 2
        else:
            n_fft = int(hparams.win_length_ms / 1000 * fs)
    else:
        n_fft = hparams.n_fft
    if rate is None:
        rate = hparams.sr
    _mel_basis = librosa.filters.mel(
        rate,
        n_fft,
        n_mels=hparams.num_mel_bins,
        fmin=hparams.mel_lower_edge_hertz,
        fmax=hparams.mel_upper_edge_hertz,
    )
    # Normalise contribution of mel coeff to 1 for 1 output bin
    return (_mel_basis.T / np.sum(_mel_basis + _epsilon(), axis=1)).T


def _normalize(S, min_db, max_db):
    return np.clip((S - min_db) / (max_db - min_db), 0, 1)


def _denormalize(S, min_db, max_db):
    return (np.clip(S, 0, 1) * (max_db - min_db)) + min_db


def _amplitude_to_db(S):
    S = np.asarray(S)
    magnitude = np.abs(S)
    power = np.square(magnitude, out=magnitude)
    return _power_to_db(power)


def _power_to_db(S):
    a_db_min = _a_min()**2
    S = np.asarray(S)
    if a_db_min <= 0:
        raise ValueError('amin < 0')
    log_spec = 10.0 * np.log10(np.maximum(a_db_min, S))
    return log_spec


def _db_to_power(S_db):
    return np.power(10.0, 0.1 * S_db)


def _db_to_amplitude(S_db):
    return _db_to_power(S_db)**0.5


def _epsilon():
    return 2e-16


def _a_min():
    return 1e-10


def _min_level_db():
    a_min_db = _a_min()**2
    min_level = 10.0 * np.log10(a_min_db)
    return min_level
