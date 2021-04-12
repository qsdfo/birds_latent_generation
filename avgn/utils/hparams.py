from avgn.utils.seconds_to_samples import sample_to_ms


class HParams(object):
    """ Hparams was removed from tf 2.0alpha so this is a placeholder
    """

    def __init__(self, **kwargs):
        self.sr = kwargs['sr']
        self.n_fft = kwargs['n_fft']
        self.win_length_samples = kwargs['win_length_samples']
        self.hop_length_samples = kwargs['hop_length_samples']
        self.ref_level_db = kwargs['ref_level_db']
        self.preemphasis = kwargs['preemphasis']
        self.num_mel_bins = kwargs['num_mel_bins']
        self.power = kwargs['power']
        self.mel_lower_edge_hertz = kwargs['mel_lower_edge_hertz']
        self.mel_upper_edge_hertz = kwargs['mel_upper_edge_hertz']
        self.butter_lowcut = kwargs['butter_lowcut']
        self.butter_highcut = kwargs['butter_highcut']
        self.reduce_noise = kwargs['reduce_noise']  # False
        self.noise_reduce_kwargs = kwargs['noise_reduce_kwargs']  # {}
        self.mask_spec = kwargs['mask_spec']  # False
        self.mask_spec_kwargs = kwargs['mask_spec_kwargs']  # {"spec_thresh": 0.9, "offset": 1e-10}
        self.n_jobs = kwargs['n_jobs']  # -1
        self.verbosity = kwargs['verbosity']  # 1
        # training chunks
        self.chunk_len_samples = kwargs['chunk_len_samples']
        if self.chunk_len_samples is not None:
            self.chunk_len_win = int((self.chunk_len_samples - self.win_length_samples) / self.hop_length_samples + 1)
            self.chunk_len_ms = sample_to_ms(self.chunk_len_samples, self.sr)
            assert self.chunk_len_win == ((self.chunk_len_samples - self.win_length_samples) / self.hop_length_samples + 1)

    def __repr__(self):
        return f'sr-{self.sr}_' \
               f'wl-{self.win_length_samples}_' \
               f'hl-{self.hop_length_samples}_' \
               f'nfft-{self.n_fft}_' \
               f'chunkls-{self.chunk_len_samples}_' \
               f'melb-{self.num_mel_bins}_' \
               f'mell-{self.mel_lower_edge_hertz}_' \
               f'melh-{self.mel_upper_edge_hertz}'
