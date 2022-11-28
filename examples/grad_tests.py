import torch

torch.set_default_dtype(torch.float64)  # we need more precision for FDTD
try:  # we don't need gradients (for now)
    torch.set_grad_enabled(True)
    print('worked')
except AttributeError:
    print('ERROR: GRAD COULD NOT BE SET TO TRUE')
    torch.set_grad_enabled(False)
print('Cuda?: ', torch.cuda.is_available())
