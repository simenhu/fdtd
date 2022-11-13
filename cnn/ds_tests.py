import numpy as np
import torch
import torchvision
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
import torchvision.transforms.functional as F
import torch.nn as nn
import torch.optim as optim


# # define how image transformed
# image_transform = torchvision.transforms.Compose([
#                                torchvision.transforms.ToTensor(),
#                                torchvision.transforms.Normalize(
#                                  (0.1307,), (0.3081,))
#                              ])
# define how image transformed
image_transform = torchvision.transforms.Compose([
                               torchvision.transforms.ToTensor()])
#image datasets
train_dataset = torchvision.datasets.CIFAR10('cifar10/', 
                                           train=True, 
                                           download=True,
                                           transform=image_transform)
#data loaders
train_loader = torch.utils.data.DataLoader(train_dataset,
                                           batch_size=64, 
                                           shuffle=True)

# We can check the dataloader
_, (example_datas, labels) = next(enumerate(train_loader))
sample = example_datas[0][0]
# show the data
plt.imshow(sample, cmap='gray', interpolation='none')
print("Label: "+ str(labels[0]))


