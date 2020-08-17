from torch import nn
import torch.nn.functional as F


class Encoder(nn.Module):
    """
    From (batch_size, num_tokens, embedding_dim)
      to (batch_size, num_tokens // prod(downscale_factors), codebook_dim)
    Uses positional embeddings
    """

    def __init__(self, n_z, conv_stack, conv2z):
        super(Encoder, self).__init__()
        self.n_z = n_z
        # Conv stack
        self.conv_stack = nn.ModuleList(conv_stack)
        # Small mlp to latent
        self.conv2z = nn.Sequential(*conv2z)

    def forward(self, x):
        # conv stack
        y = x
        for layer in range(len(self.conv_stack)):
            y = self.conv_stack[layer](y)
            y = F.relu(y)
        # flattening
        y = y.view(y.size(0), -1)
        # map to latent dim
        z = self.conv2z(y)
        return z
