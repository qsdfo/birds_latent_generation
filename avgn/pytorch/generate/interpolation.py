from avgn.signalprocessing.spectrogramming import build_mel_basis, build_mel_inversion_basis, inv_spectrogram_librosa
from avgn.utils.cuda_variable import cuda_variable
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf


def plot_interpolations(model, hparams, dataloader, savepath, num_interpolated_points, method):
    # Forward pass
    model.eval()
    for _, data in enumerate(dataloader):
        x_cuda = cuda_variable(data['input'])
        # Get z
        mu, logvar = model.encode(x_cuda)
        z = model.reparameterize(mu, logvar)
        # Arbitrarily choose start and end points as batch_ind and batch_ind + 1
        start_z = z[:-1]
        end_z = z[1:]
        batch_dim, rgb_dim, h_dim, w_dim = x_cuda.shape
        num_examples = batch_dim - 1
        x_interpolation = np.zeros((num_examples, rgb_dim, h_dim, w_dim, num_interpolated_points))

        ind_interp = 0
        for t in np.linspace(start=0, stop=1, num=num_interpolated_points):
            # Perform interp
            if method == 'linear':
                this_z = start_z * (1 - t) + end_z * t
            else:
                raise NotImplemented
            # Decode z
            x_recon = model.decode(this_z).cpu().detach().numpy()
            x_interpolation[:, :, :, :, ind_interp] = x_recon
            ind_interp = ind_interp + 1
        break

    # Plot
    dims = h_dim, w_dim
    plt.clf()
    fig, axes = plt.subplots(nrows=num_examples, ncols=num_interpolated_points)
    for ind_example in range(num_examples):
        for ind_interp in range(num_interpolated_points):
            # show the image
            axes[ind_example, ind_interp].matshow(x_interpolation[ind_example, :, :, :, ind_interp].reshape(dims),
                                                  origin="lower")
    for ax in fig.get_axes():
        ax.set_xticks([])
        ax.set_yticks([])
    plt.savefig(f'{savepath}/spectro.pdf')
    plt.close('all')

    # audio
    if hparams is not None:
        mel_basis = build_mel_basis(hparams, hparams.sr, hparams.sr)
        mel_inversion_basis = build_mel_inversion_basis(mel_basis)
        for ind_example in range(num_examples):
            for ind_interp in range(num_interpolated_points):
                audio = inv_spectrogram_librosa(x_interpolation[ind_example, 0, :, :, ind_interp], hparams.sr, hparams,
                                                mel_inversion_basis=mel_inversion_basis)
                sf.write(f'{savepath}/{ind_example}_{ind_interp}.wav', audio, samplerate=hparams.sr)
