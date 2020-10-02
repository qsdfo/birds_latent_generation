import math

from avgn.signalprocessing.spectrogramming import build_mel_basis, build_mel_inversion_basis, inv_spectrogram_librosa
from avgn.utils.cuda_variable import cuda_variable
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
import torch


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
            elif method == 'constant_radius':
                this_z = constant_radius_interpolation(start_z, end_z, t)
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
    audios = None
    if hparams is not None:
        mel_basis = build_mel_basis(hparams, hparams.sr, hparams.sr)
        mel_inversion_basis = build_mel_inversion_basis(mel_basis)
        for ind_example in range(num_examples):
            for ind_interp in range(num_interpolated_points):
                audio = inv_spectrogram_librosa(x_interpolation[ind_example, 0, :, :, ind_interp], hparams.sr, hparams,
                                                mel_inversion_basis=mel_inversion_basis)
                if audios is None:
                    audios = np.zeros((num_examples, num_interpolated_points, len(audio)))
                audios[ind_example, ind_interp] = audio
                sf.write(f'{savepath}/{ind_example}_{ind_interp}.wav', audio, samplerate=hparams.sr)
    return {
        'audios': audios,
        'spectros': x_interpolation,
        'dims': dims
    }


def constant_radius_interpolation(start_z, end_z, t):
    # spherical
    start_z_sph = cartesian_to_spherical(start_z)
    end_z_sph = cartesian_to_spherical(end_z)
    # interp
    this_z_sph = start_z_sph * (1 - t) + end_z_sph * t
    # back to cartesian
    this_z = spherical_to_cartesian(this_z_sph)
    # Notes: checked start_z ~ this_z when t = 0, with small enough numerical errors
    return this_z


def cartesian_to_spherical(z):
    z_shape = z.shape
    assert len(z_shape) == 2
    batch_size, dim = z_shape
    s = torch.zeros_like(z)
    s[:, 0] = torch.sqrt(torch.sum(z ** 2, dim=1))
    for i in range(dim - 2):
        norm_term = torch.sqrt(torch.sum(z[:, i:] ** 2, dim=1))
        term = torch.where(torch.eq(norm_term, 0),
                           torch.zeros_like((s[:, 0])),
                           torch.acos(z[:, i] / norm_term)
                           )
        s[:, i + 1] = term
    # last term
    norm_term = torch.sqrt(torch.sum(z[:, dim - 2:] ** 2, dim=1))
    acos_term = torch.where(torch.eq(norm_term, 0),
                            torch.zeros_like((s[:, 0])),
                            torch.acos(z[:, dim - 2] / norm_term))
    last_term = torch.where(torch.ge(z[:, -1], 0),
                            acos_term,
                            2 * math.pi - acos_term
                            )
    s[:, -1] = last_term
    return s


def spherical_to_cartesian(s):
    s_shape = s.shape
    assert len(s_shape) == 2
    batch_size, dim = s_shape
    z = torch.zeros_like(s)
    radius = s[:, 0]
    phis = s[:, 1:]
    for i in range(dim-1):
        if i > 0:
            sin_term = torch.prod(torch.sin(phis[:, :i-1]), dim=1)
        else:
            sin_term = 1
        z[:, i] = radius * torch.cos(phis[:, i]) * sin_term
    # Last term
    z[:, -1] = radius * torch.prod(torch.sin(phis), dim=1)
    return z
