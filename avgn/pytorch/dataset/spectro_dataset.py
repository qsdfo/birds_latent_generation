import random

import librosa
from avgn.signalprocessing.spectrogramming_scipy import build_mel_basis, spectrogram_sp
import pickle
import numpy as np
import torch
from torch.utils.data import Dataset
import pyrubberband


class SpectroDataset(Dataset):
    def __init__(self, syllable_paths, data_processing, data_augmentations):
        super(SpectroDataset, self).__init__()
        self.syllable_paths = syllable_paths
        self.data_processing = data_processing
        self.mel_basis = build_mel_basis(data_processing['n_fft'],
                                         data_processing['sr'],
                                         data_processing['num_mel_bins'],
                                         data_processing['mel_lower_edge_hertz'],
                                         data_processing['mel_upper_edge_hertz'],
                                         )
        self.data_augmentations = data_augmentations

    def __len__(self):
        return len(self.syllable_paths)

    @staticmethod
    def process_mSp(mSp):
        mSp_np = np.array(mSp)
        if mSp_np.max() <= 1:
            x_np = mSp_np.astype(np.float32)
        else:
            x_np = mSp_np.astype(np.float32) / 255.
        return np.expand_dims(x_np, axis=0)

    def __getitem__(self, idx):
        import time
        aaa = time.time()
        if torch.is_tensor(idx):
            idx = idx.tolist()
        fname = self.syllable_paths[idx]
        with open(fname, 'rb') as ff:
            data = pickle.load(ff)

        sn = data['sn']

        # data augmentations
        aaa = time.time()
        # if self.data_augmentations:
        #     time_shift = random.uniform(0.75, 1.25)
        #     sn_t = pyrubberband.pyrb.time_stretch(
        #         sn, sr=self.data_processing['sr'], rate=time_shift)
        #     pitch_shift = random.randrange(-2, 2)
        #     sn_tp = pyrubberband.pyrb.pitch_shift(
        #         sn_t, sr=self.data_processing['sr'], n_steps=pitch_shift)
        # else:
        #     sn_tp = sn
        if self.data_augmentations:
            time_shift = random.uniform(0.75, 1.25)
            sn_t = librosa.effects.time_stretch(y=sn, rate=time_shift)
            pitch_shift = random.randrange(-4, 4)
            sn_tp = librosa.effects.pitch_shift(y=sn_t, sr=self.data_processing['sr'],
                                                n_steps=pitch_shift, bins_per_octave=24)
        else:
            sn_tp = sn
        bbb = time.time()
        print(f'data aug {bbb-aaa}')

        # create spec
        aaa = time.time()
        mSp, _ = spectrogram_sp(y=sn_tp,
                                sr=self.data_processing['sr'],
                                n_fft=self.data_processing['n_fft'],
                                win_length=self.data_processing['win_length'],
                                hop_length=self.data_processing['hop_length'],
                                ref_level_db=self.data_processing['ref_level_db'],
                                _mel_basis=self.mel_basis,
                                pre_emphasis=self.data_processing['preemphasis'],
                                power=self.data_processing['power'],
                                debug=True
                                )
        bbb = time.time()
        print(f'spectrogramming {bbb-aaa}')

        # pad
        aaa = time.time()
        win_len = mSp.shape[1]
        if win_len < self.data_processing['chunk_len_win']:
            pad_size = (self.data_processing['chunk_len_win'] - win_len) // 2
            mSp_pad = np.zeros(
                (self.data_processing['num_mel_bins'], self.data_processing['chunk_len_win']))
            mSp_pad[:, pad_size:(pad_size + win_len)] = mSp
        else:
            mSp_pad = mSp[:, :self.data_processing['chunk_len_win']]
        bbb = time.time()
        print(f'padding {bbb-aaa}')

        # conv in pytorch are
        # (batch, channel, height, width)
        sample = SpectroDataset.process_mSp(mSp_pad)

        ########################################################################
        ########################################################################
        # DEBUG
        # plot everything
        # import soundfile as sf
        # import os
        # import matplotlib.pyplot as plt
        # mel_inversion_basis = build_mel_inversion_basis(self.mel_basis)
        # dump_folder = 'dump/batch'
        # if not os.path.isdir(dump_folder):
        #     os.makedirs(dump_folder)
        # sf.write(f'{dump_folder}/{idx}_sn.wav', sn, samplerate=self.data_processing['sr'])
        # sf.write(f'{dump_folder}/{idx}_sn_tp.wav',
        #          sn_tp, samplerate=self.data_processing['sr'])
        # plt.clf()
        # plt.matshow(mSp_pad, origin="lower")
        # plt.savefig(f'{dump_folder}/{idx}_mS.pdf')
        # plt.close()
        # audio_reconstruct = inv_spectrogram_sp(mSp_pad, n_fft=self.data_processing['n_fft'],
        #                                        win_length=self.data_processing['win_length_samples'],
        #                                        hop_length=self.data_processing['hop_length_samples'],
        #                                        ref_level_db=self.data_processing['ref_level_db'],
        #                                        power=self.data_processing['power'],
        #                                        mel_inversion_basis=mel_inversion_basis)
        # sf.write(f'{dump_folder}/{idx}_mS.wav',
        #          audio_reconstruct, samplerate=self.data_processing['sr'])
        ########################################################################
        ########################################################################
        return {
            'input': sample,
            'target': sample,
            'label': data['label']
        }
