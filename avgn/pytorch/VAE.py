import torch
from torch import nn
from torch.nn import functional as F


class VAE(nn.Module):
    """a basic vae class for tensorflow
    Extends:
        tf.keras.Model
    """

    def __init__(self, encoder, decoder, beta=1.0):
        super(VAE, self).__init__()
        self.beta = beta
        self.enc = encoder
        self.dec = decoder
        self.n_z = encoder.n_z

    def save(self, path):
        torch.save(self.state_dict(), f'{path}/model')

    def load(self, path, device):
        self.load_state_dict(torch.load(f'{path}/model', map_location=torch.device(device)))

    def save_model(self, path):
        self.enc.save_weights(f'{path}/encoder.h5')
        self.dec.save_weights(f'{path}/decoder.h5')

    def load_model(self, path):
        self.enc.load_weights(f'{path}/encoder.h5')
        self.dec.load_weights(f'{path}/decoder.h5')

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

    def step(self, x):
        recon_batch, mu, logvar = self(x)
        batch_dim = x.shape[0]
        recon_batch_flat = recon_batch.view(batch_dim, -1)
        x_flat = x.view(batch_dim, -1)
        loss = self.loss_function(recon_batch_flat, x_flat, mu, logvar, self.beta)
        return loss

    # Reconstruction + KL divergence losses summed over all elements and batch
    @staticmethod
    def loss_function(recon_x, x, mu, logvar, beta):
        BCE = F.binary_cross_entropy(recon_x, x, reduction='sum')
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return BCE + beta * KLD

