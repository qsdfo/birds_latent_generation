from avgn.signalprocessing.spectrogramming_scipy import build_mel_basis, build_mel_inversion_basis, inv_spectrogram_sp
import matplotlib.pyplot as plt
import soundfile as sf


def plot_generation(model, hparams, num_examples, savepath):
    # forward pass
    model.eval()
    gen = model.generate(batch_dim=num_examples).cpu().detach().numpy()

    # plot
    dims = gen.shape[2:]
    plt.clf()
    fig, axes = plt.subplots(ncols=num_examples)
    for i in range(num_examples):
        # show the image
        axes[i].matshow(gen[i].reshape(dims), origin="lower")
    for ax in fig.get_axes():
        ax.set_xticks([])
        ax.set_yticks([])
    plt.savefig(f'{savepath}/spectro.pdf')
    plt.close('all')

    # audio
    audios = []
    if hparams is not None:
        mel_basis = build_mel_basis(hparams, hparams.sr, hparams.sr)
        mel_inversion_basis = build_mel_inversion_basis(mel_basis)
        for i in range(num_examples):
            gen_audio = inv_spectrogram_sp(gen[i, 0],
                                           n_fft=hparams.n_fft,
                                           win_length=hparams.win_length_samples,
                                           hop_length=hparams.hop_length_samples,
                                           ref_level_db=hparams.ref_level_db,
                                           power=hparams.power,
                                           mel_inversion_basis=mel_inversion_basis
                                           )
            audios.append(gen_audio)
            sf.write(f'{savepath}/{i}.wav', gen_audio, samplerate=hparams.sr)
    return {
        'audios': audios,
        'spectros': gen
    }
