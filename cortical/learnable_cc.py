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
from torch.utils.tensorboard import SummaryWriter
from autoencoder import AutoEncoder

# Writer will output to ./runs/ directory by default
writer = SummaryWriter()

# ## Set Backend
#backend_name = "torch.cuda.float64"
backend_name = "torch"
fdtd.set_backend(backend_name)
if(backend_name.startswith("torch.cuda")):
    device = "cuda"
else:
    device = "cpu"


# ## Constants
WAVELENGTH = 1550e-9
SPEED_LIGHT: float = 299_792_458.0  # [m/s] speed of light


# ## Simulation

# create FDTD Grid

# In[4]:


grid = fdtd.Grid(
    (52, 52, 1),
    grid_spacing=0.1 * WAVELENGTH,
    permittivity=1.0,
    permeability=1.0,
)

print('Grid Shape: ', grid.shape)


# Boundaries with width bw
bw = 10
grid[  0: bw, :, :] = fdtd.LearnableAnisotropicObject(permittivity=2.5, name="xlow")
grid[-bw:   , :, :] = fdtd.LearnableAnisotropicObject(permittivity=2.5, name="xhigh")

grid[:,   0:bw, :] = fdtd.LearnableAnisotropicObject(permittivity=2.5, name="ylow")
grid[:, -bw:  , :] = fdtd.LearnableAnisotropicObject(permittivity=2.5, name="yhigh")

grid[:, :, 0] = fdtd.PeriodicBoundary(name="zbounds")



# sources

#TODO make sure this source covers enough of the grid
grid[10:42,10:42,0] = fdtd.CorticalColumnPlaneSource(
    period = WAVELENGTH / SPEED_LIGHT,
    polarization = 'x', # BS value, polarization is not used.
    name='cc'
)

# grid[35,35,0] = fdtd.PointSource(
#     period = WAVELENGTH / SPEED_LIGHT,
#     name='ps'
# )

# Object defining the cortical column substrate 
grid[bw:-bw, bw:-bw, :] = fdtd.LearnableAnisotropicObject(permittivity=2.5, name="learnable_object")


# Make the model
model = AutoEncoder(grid=grid, input_chans=1, output_chans=1).to(device)

print('Get object: ', [obj.name for obj in grid.objects])
params_to_learn = [obj.inverse_permittivity for obj in grid.objects]
params_to_learn += [*model.parameters()]
#learning_rate = 0.00001
learning_rate = 0.01
#learning_rate = 0.1
#learning_rate = 1.0
optimizer = optim.SGD(params_to_learn, lr=learning_rate,
                      momentum=0.5)
mse = torch.nn.MSELoss(reduce=False)

max_train_steps = 1000000000000000
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
    if(train_step % 100 == 0):
        vis = True
    else:
        vis = False
    y = model(img, em_steps, visualize=vis)
    # Add images to tensorboard
    img_grid = torchvision.utils.make_grid([img[0,...], y])
    writer.add_image('images', img_grid, train_step)
    ### X ### - Generate loss
    loss = loss_fn(y, img)
    writer.add_scalar('Loss', loss, train_step)
    print('Train step: ', train_step, '\tTime: ', grid.time_steps_passed, '\tLoss: ', loss)
    print('Model cc_dirs: ', torch.sum(model.cc_dirs)) 
    print('Model cc_freqs: ', torch.sum(model.cc_freqs))
    print('Model cc_phases: ', torch.sum(model.cc_phases))
    writer.add_histogram('cc_dirs', model.cc_dirs, train_step)
    writer.add_histogram('cc_freqs', model.cc_freqs, train_step)
    writer.add_histogram('cc_phases', model.cc_phases, train_step)

    optimizer.zero_grad()
    ### X ### - Backprop
    loss.backward(retain_graph=True)
    optimizer.step()
    counter += 1
    # if(train_step % 20 == 0):
    #     img_to_show = torch.permute(y, (1,2,0)).detach().cpu()
    #     plt.imshow(img_to_show, cmap='gray', interpolation='none')
    #     plt.show()


writer.close()
