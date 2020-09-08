from avgn.signalprocessing.spectrogramming import build_mel_basis, build_mel_inversion_basis, inv_spectrogram_librosa
from avgn.utils.cuda_variable import cuda_variable
import matplotlib.pyplot as plt
import soundfile as sf


def plot_reconstruction(model, hparams, dataloader, savepath):
    # Forward pass
    model.eval()
    for _, data in enumerate(dataloader):
        x_cuda = cuda_variable(data['input'])
        x_recon = model.reconstruct(x_cuda).cpu().detach().numpy()
        break
    # Plot
    x_orig = data['input'].numpy()
    dims = x_recon.shape[2:]
    num_examples = x_recon.shape[0]
    plt.clf()
    fig, axes = plt.subplots(nrows=2, ncols=num_examples)
    for i in range(num_examples):
        # show the image
        axes[0, i].matshow(x_orig[i].reshape(dims), origin="lower")
        axes[1, i].matshow(x_recon[i].reshape(dims), origin="lower")
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
            original_audio = inv_spectrogram_librosa(x_orig[i, 0], hparams.sr, hparams,
                                                     mel_inversion_basis=mel_inversion_basis)
            recon_audio = inv_spectrogram_librosa(x_recon[i, 0], hparams.sr, hparams,
                                                  mel_inversion_basis=mel_inversion_basis)
            sf.write(f'{savepath}/{i}_original.wav', original_audio, samplerate=hparams.sr)
            sf.write(f'{savepath}/{i}_recon.wav', recon_audio, samplerate=hparams.sr)
