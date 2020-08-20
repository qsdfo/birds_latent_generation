import torch

def cuda_variable(tensor, non_blocking=False):
    if torch.cuda.is_available():
        return tensor.to('cuda', non_blocking=non_blocking)
    else:
        return tensor