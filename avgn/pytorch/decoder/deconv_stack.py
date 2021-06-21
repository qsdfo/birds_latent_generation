from torch import nn
import numpy as np
import torch
import torch.nn.functional as F


class Deconv_stack(nn.Module):
    """
    From (batch_size, num_tokens, embedding_dim)
      to (batch_size, num_tokens // prod(downscale_factors), codebook_dim)
    Uses positional embeddings
    """

    def __init__(self, deconv_input_shape, z2deconv, deconv_stack, n_z):
        super(Deconv_stack, self).__init__()
        self.n_z = n_z
        self.deconv_input_shape = deconv_input_shape
        self.z2deconv = nn.Sequential(*z2deconv)
        self.deconv_stack = nn.ModuleList(deconv_stack)
        return

    def forward(self, z):
        # ffnn
        y = self.z2deconv(z)
        y = y.view(-1, *self.deconv_input_shape)

        # deconvolution stack
        for layer in range(len(self.deconv_stack) - 1):
            y = self.deconv_stack[layer](y)
            y = F.relu(y)
        y = self.deconv_stack[-1](y)
        x = torch.sigmoid(y)
        return x
