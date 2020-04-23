from torch import nn
import numpy as np
import torch.nn.functional as F


class Decoder(nn.Module):
    """
    From (batch_size, num_tokens, embedding_dim)
      to (batch_size, num_tokens // prod(downscale_factors), codebook_dim)
    Uses positional embeddings
    """

    def __init__(self, deconv_input_shape, deconv_stack, n_z):
        super(Decoder, self).__init__()
        self.deconv_input_shape = deconv_input_shape
        prod_deconv_input_shape = int(np.prod(deconv_input_shape))
        self.to_deconv_stack = nn.Sequential(
            nn.Linear(in_features=n_z, out_features=prod_deconv_input_shape),
            nn.ReLU()
        )
        self.deconv_stack = nn.ModuleList(deconv_stack)
        return

    def forward(self, z):
        """
        """
        y = self.to_deconv_stack(z)
        y = y.view(-1, *self.deconv_input_shape)
        # deconvolution stack
        for layer in range(len(self.deconv_stack)-1):
            y = self.deconv_stack[layer](y)
            y = F.relu(y)
        y = self.deconv_stack[-1](y)
        x = F.sigmoid(y)
        return x
