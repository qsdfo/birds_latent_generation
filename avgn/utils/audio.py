import wave

import numpy as np
from scipy.io import wavfile


def write_wav(loc, rate, data):
    wavfile.write(loc, rate, data)


def int16_to_float32(data):
    """ Converts from uint16 wav to float32 wav
    """
    if np.max(np.abs(data)) > 32768:
        raise ValueError("Data has values above 32768")
    return (data / 32768.0).astype("float32")


def float32_to_uint8(data):
    """ convert from float32 to uint8 (256)
    """
    raise NotImplementedError


def float32_to_int16(data):
    if np.max(data) > 1:
        data = data / np.max(np.abs(data))
    return np.array(data * 32767).astype("int16")


def get_samplerate(file):
    with wave.open(file, "rb") as f:
        samplerate = f.getframerate()
    return samplerate
