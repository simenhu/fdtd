import torch

def get_default_device(index=0):
    """Get the default device depending on cuda availability"""

    if torch.cuda.is_available():
        devices = [index for index in range(torch.cuda.device_count())]
        return torch.device(f'cuda:{devices[index]}')
    else:
        return torch.device('cpu')