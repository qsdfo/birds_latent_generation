from torch import nn
import numpy as np
import torch.nn.functional as F


class Decoder(nn.Module):
    """
    From (batch_size, num_tokens, embedding_dim)
      to (batch_size, num_tokens // prod(downscale_factors), codebook_dim)
    Uses positional embeddings
    """

    def __init__(self, deconv_input_shape, n_z):
        self.deconv_input_shape = deconv_input_shape
        prod_deconv_input_shape = int(np.prod(deconv_input_shape))
        super(Decoder, self).__init__()
        self.to_deconv_stack = nn.Sequential(
            nn.Linear(in_features=n_z, out_features=prod_deconv_input_shape),
            nn.ReLU()
        )
        self.deconv1 = nn.ConvTranspose2d(deconv_input_shape[0], 64, (4, 2), stride=(4, 2))  # (b, 64, 32, 12)
        self.deconv2 = nn.ConvTranspose2d(64, 32, (4, 2), stride=(4, 2))  # (b, 32, 128, 24)
        self.deconv3 = nn.ConvTranspose2d(32, 1, 1, stride=(1, 1))
        return

    def forward(self, z):
        """

        :param inputs: (batch, seq_len, dim)
        with seq = num_blocks * block_size
        :return: z: (batch,
        """
        y = self.to_deconv_stack(z)
        y = y.view(-1, *self.deconv_input_shape)
        # deconvolution stack
        y = self.deconv1(y)
        y = F.relu(y)
        y = self.deconv2(y)
        y = F.relu(y)
        y = self.deconv3(y)
        x = F.sigmoid(y)
        return x
