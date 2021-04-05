def ms_to_sample(time_ms, sr):
    return int(time_ms * sr / 1000)

def sample_to_ms(samples, sr):
    return samples * 1000 / sr
