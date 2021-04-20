import random
from avgn.signalprocessing.spectrogramming_scipy import build_mel_basis, build_mel_inversion_basis, inv_spectrogram_sp
from avgn.signalprocessing.spectrogramming_scipy import spectrogram_sp
import pickle
import numpy as np
import torch
from torch.utils.data import Dataset
import pyrubberband


class SpectroDataset(Dataset):
    def __init__(self, syllable_paths, hparams, data_augmentations):
        super(SpectroDataset, self).__init__()
        self.syllable_paths = syllable_paths
        self.hparams = hparams
        self.mel_basis = build_mel_basis(hparams, hparams.sr, hparams.sr)
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
        if torch.is_tensor(idx):
            idx = idx.tolist()
        fname = self.syllable_paths[idx]
        with open(fname, 'rb') as ff:
            data = pickle.load(ff)

        sn = data['sn']

        if self.data_augmentations:
            # data augmentations
            time_shift = random.uniform(0.75, 1.25)
            sn_t = pyrubberband.pyrb.time_stretch(
                sn, sr=self.hparams.sr, rate=time_shift)
            pitch_shift = random.randrange(-2, 2)
            sn_tp = pyrubberband.pyrb.pitch_shift(
                sn_t, sr=self.hparams.sr, n_steps=pitch_shift)

        # create spec
        mSp, _ = spectrogram_sp(y=sn_tp,
                                sr=self.hparams.sr,
                                n_fft=self.hparams.n_fft,
                                win_length=self.hparams.win_length_samples,
                                hop_length=self.hparams.hop_length_samples,
                                ref_level_db=self.hparams.ref_level_db,
                                _mel_basis=self.mel_basis,
                                pre_emphasis=self.hparams.preemphasis,
                                power=self.hparams.power,
                                debug=True
                                )

        # pad
        win_len = mSp.shape[1]
        if win_len < self.hparams.chunk_len_win:
            pad_size = (self.hparams.chunk_len_win - win_len) // 2
            mSp_pad = np.zeros(
                (self.hparams.num_mel_bins, self.hparams.chunk_len_win))
            mSp_pad[:, pad_size:(pad_size + win_len)] = mSp
        else:
            mSp_pad = mSp[:, :self.hparams.chunk_len_win]

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
        # sf.write(f'{dump_folder}/{idx}_sn.wav', sn, samplerate=self.hparams.sr)
        # sf.write(f'{dump_folder}/{idx}_sn_tp.wav',
        #          sn_tp, samplerate=self.hparams.sr)
        # plt.clf()
        # plt.matshow(mSp_pad, origin="lower")
        # plt.savefig(f'{dump_folder}/{idx}_mS.pdf')
        # plt.close()
        # audio_reconstruct = inv_spectrogram_sp(mSp_pad, n_fft=self.hparams.n_fft,
        #                                        win_length=self.hparams.win_length_samples,
        #                                        hop_length=self.hparams.hop_length_samples,
        #                                        ref_level_db=self.hparams.ref_level_db,
        #                                        power=self.hparams.power,
        #                                        mel_inversion_basis=mel_inversion_basis)
        # sf.write(f'{dump_folder}/{idx}_mS.wav',
        #          audio_reconstruct, samplerate=self.hparams.sr)
        ########################################################################
        ########################################################################
        return {
            'input': sample,
            'target': sample,
            'label': data['label']
        }
