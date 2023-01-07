#!/usr/bin/env python

# Here we demonstrate learning cortical columns.

import sys
sys.path.append('/home/bij/Projects/fdtd/')
import fdtd
import fdtd.backend as bd
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.optim as optim
import torchvision
from autoencoder import AutoEncoder


# ## Set Backend
fdtd.set_backend("torch")


# ## Constants
WAVELENGTH = 1550e-9
WAVELENGTH2 = 1550e-8
SPEED_LIGHT: float = 299_792_458.0  # [m/s] speed of light


# ## Simulation

# create FDTD Grid

# In[4]:


grid = fdtd.Grid(
    (1.5e-5, 1.5e-5, 1),
    grid_spacing=0.1 * WAVELENGTH,
    permittivity=1.0,
    permeability=1.0,
)

print('Grid Shape: ', grid.shape)


# boundaries - make them learnable objects

# For some reason these don't reset properly.
grid[0:10, :, :] = fdtd.LearnableAnisotropicObject(permittivity=2.5, name="xlow")
grid[-10:, :, :] = fdtd.LearnableAnisotropicObject(permittivity=2.5, name="xhigh")

grid[:, 0:10, :] = fdtd.LearnableAnisotropicObject(permittivity=2.5, name="ylow")
grid[:, -10:, :] = fdtd.LearnableAnisotropicObject(permittivity=2.5, name="yhigh")

grid[:, :, 0] = fdtd.PeriodicBoundary(name="zbounds")



# sources

gl = grid.shape[0]

# grid[gl//2,gl//5,0] = fdtd.PointSource(
#     period = WAVELENGTH / SPEED_LIGHT,
# )

#TODO make sure polarization makes sense
#TODO make sure this source covers enough of the grid
grid[20:52,20:52,0] = fdtd.CorticalColumnPlaneSource(
    period = WAVELENGTH / SPEED_LIGHT,
    polarization = 'x',
    name='cc'
)

# detectors

# objects

midpoint_y = grid.shape[0]//2
midpoint_x = grid.shape[1]//2
#grid[10:gl-10, midpoint_x-10:midpoint_x+10, 0:1] = fdtd.LearnableAnisotropicObject(permittivity=2.5, name="learnable_object")


# Add a detector
#my_detector = fdtd.LineDetector(name="detector")
#grid[midpoint_y-3:midpoint_y+3, midpoint_x+30, 0:1] = my_detector



momentum = 0.5
device = "cuda"
# Make the model
model = AutoEncoder(grid=grid, input_chans=1, output_chans=1).to(device)

print('Get object: ', [obj.name for obj in grid.objects])
params_to_learn = [obj.inverse_permittivity for obj in grid.objects]
params_to_learn += [*model.parameters()]
#learning_rate = 0.00001
learning_rate = 0.01
optimizer = optim.SGD(params_to_learn, lr=learning_rate,
                      momentum=momentum)
mse = torch.nn.MSELoss(reduce=False)

max_train_steps = 100000
em_steps = 200 

image_transform = torchvision.transforms.Compose([
                               torchvision.transforms.ToTensor()])
train_dataset = torchvision.datasets.CIFAR10('cifar10/', 
                                           train=True, 
                                           download=True,
                                           transform=image_transform)
#data loaders
train_loader = torch.utils.data.DataLoader(train_dataset,
                                           batch_size=1, 
                                           shuffle=True)


def get_sample_img(img_loader):
    _, (example_datas, labels) = next(enumerate(img_loader))
    sample = example_datas[0][0]
    sample = sample.to(device)[None, None, :]
    return sample

sample = get_sample_img(train_loader)
print('Sample shape: ', sample.shape)
# show the data
plt.imshow(sample.cpu()[0,0,...], cmap='gray', interpolation='none')

grid.H.requires_grad = True
grid.H.retain_grad()
grid.E.requires_grad = True
grid.E.retain_grad()

loss_fn = torch.nn.MSELoss()

# Train the weights
counter = 0
print('Sum of perm: ', bd.sum(grid.objects[0].inverse_permittivity))
for train_step in range(max_train_steps):
    grid.reset()
    optimizer.zero_grad()
    ### X ### - Get a sample from training data
    img = get_sample_img(train_loader)
    ### X ### - Push it through Encoder
    y = model(img, em_steps, visualize=True)
    ### X ### - Generate loss
    loss = loss_fn(y, img)
    print('Train step: ', train_step, '\tTime: ', grid.time_steps_passed, '\tLoss: ', loss)
    print('Model cc_dirs: ', torch.sum(model.cc_dirs)) 
    print('Model cc_freqs: ', torch.sum(model.cc_freqs))
    print('Model cc_phases: ', torch.sum(model.cc_phases))
    optimizer.zero_grad()
    ### X ### - Backprop
    loss.backward(retain_graph=True)
    optimizer.step()
    counter += 1
    grid.visualize(z=0, norm='log', animate=True)
    plt.show()


