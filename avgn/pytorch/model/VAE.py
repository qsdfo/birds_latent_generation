import os

import torch
from torch import nn
from torch.nn import functional as F

from avgn.utils.cuda_variable import cuda_variable


class VAE(nn.Module):
    """a basic vae class for tensorflow
    Extends:
        tf.keras.Model
    """

    def __init__(self, encoder, decoder, model_dir, beta=1.0):
        super(VAE, self).__init__()
        self.beta = beta
        self.enc = encoder
        self.dec = decoder
        self.n_z = encoder.n_z
        self.model_dir = model_dir

    def save(self, name):
        if not os.path.isdir(f'{self.model_dir}/weights'):
            os.mkdir(f'{self.model_dir}/weights')
        torch.save(self.state_dict(), f'{self.model_dir}/weights/{name}')

    def load(self, name, device):
        self.load_state_dict(torch.load(f'{self.model_dir}/weights/{name}', map_location=torch.device(device)))

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
        return self.decode(z), mu, logvar

    def step(self, data):
        x = cuda_variable(data['input'])
        recon_batch, mu, logvar = self(x)
        batch_dim = x.shape[0]
        recon_batch_flat = recon_batch.view(batch_dim, -1)
        x_flat = x.view(batch_dim, -1)
        loss = self.loss_function(recon_batch_flat, x_flat, mu, logvar, self.beta)
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
    def loss_function(recon_x, x, mu, logvar, beta):
        BCE = F.binary_cross_entropy(recon_x, x, reduction='sum')
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return BCE + beta * KLD
