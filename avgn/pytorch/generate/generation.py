from avgn.signalprocessing.spectrogramming import build_mel_basis, build_mel_inversion_basis, inv_spectrogram_librosa
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
    if hparams is not None:
        mel_basis = build_mel_basis(hparams, hparams.sr, hparams.sr)
        mel_inversion_basis = build_mel_inversion_basis(mel_basis)
        for i in range(num_examples):
            gen_audio = inv_spectrogram_librosa(gen[i, 0], hparams.sr, hparams, mel_inversion_basis=mel_inversion_basis)
            sf.write(f'{savepath}/{i}.wav', gen_audio, samplerate=hparams.sr)