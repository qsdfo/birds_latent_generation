from torch import nn


class Encoder(nn.Module):
    """
    From (batch_size, num_tokens, embedding_dim)
      to (batch_size, num_tokens // prod(downscale_factors), codebook_dim)
    Uses positional embeddings
    """

    def __init__(self, input_dim, n_z):
        h_dim, w_dim = input_dim
        super(Encoder, self).__init__()
        self.convolution_stack = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=(3, 3)),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=(3, 3)),
            nn.ReLU(),
        )

        in_features = h_dim * w_dim / (3 * 3 * 64)
        self.z_mapping = nn.Sequential(
            nn.Linear(in_features=in_features, out_features=n_z*2),
        )

    def forward(self, x):
        """

        :param inputs: (batch, seq_len, dim)
        with seq = num_blocks * block_size
        :return: z: (batch,
        """
        # conv stack
        y = self.convolution_stack(x)
        # flattening
        y = y.view(y.size(0), -1)
        # map to latent dim
        z = self.z_mapping(y)
        return z
