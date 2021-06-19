import os
import torch
import torch.nn as nn

from avgn.utils.cuda_variable import cuda_variable


class VAE_MMD(nn.Module):
    """a basic vae class for tensorflow
    Extends:
        tf.keras.Model
    """

    def __init__(self, encoder, decoder, model_dir):
        super(VAE_MMD, self).__init__()
        self.enc = encoder
        self.dec = decoder
        self.n_z = encoder.n_z
        self.model_dir = model_dir

    def save(self, name):
        if not os.path.isdir(f'{self.model_dir}/weights'):
            os.mkdir(f'{self.model_dir}/weights')
        torch.save(self.state_dict(), f'{self.model_dir}/weights/{name}')

    def load(self, name, device):
        self.load_state_dict(torch.load(
            f'{self.model_dir}/weights/{name}', map_location=torch.device(device)))

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def encode(self, x):
        x_enc = self.enc(x)
        mu = x_enc[:, :self.n_z]
        logvar = x_enc[:, self.n_z:]
        return mu, logvar

    def decode(self, z):
        return self.dec(z)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), z

    def step(self, data):
        x = cuda_variable(data['input'])
        recon_batch, z = self(x)
        batch_dim = x.shape[0]
        recon_batch_flat = recon_batch.view(batch_dim, -1)
        x_flat = x.view(batch_dim, -1)
        loss = self.loss_mmd(recon_batch_flat, x_flat, z)
        return loss

    def reconstruct(self, x):
        return self(x)[0]

    def generate(self, batch_dim):
        # Prior is a gaussian w/ mean 0 and variance 1
        z = cuda_variable(torch.randn(batch_dim, self.n_z))
        x = self.decode(z)
        return x

    # Reconstruction + KL divergence losses summed over all elements and batch
    @staticmethod
    def loss_mmd(recon_x, x, z):
        loss = (recon_x-x).pow(2).mean() + MMD(torch.randn(200,
                                                          z.size(1), requires_grad=False).to(x), z)
        return loss


def gaussian_kernel(a, b):
    dim1_1, dim1_2 = a.shape[0], b.shape[0]
    depth = a.shape[1]
    a = a.view(dim1_1, 1, depth)
    b = b.view(1, dim1_2, depth)
    a_core = a.expand(dim1_1, dim1_2, depth)
    b_core = b.expand(dim1_1, dim1_2, depth)
    numerator = (a_core - b_core).pow(2).mean(2)/depth
    return torch.exp(-numerator)


def MMD(a, b):
    return gaussian_kernel(a, a).mean() + gaussian_kernel(b, b).mean() - 2*gaussian_kernel(a, b).mean()
