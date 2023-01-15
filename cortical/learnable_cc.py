#!/usr/bin/env python

# Here we demonstrate learning cortical columns.

import sys
sys.path.append('/home/bij/Projects/fdtd/')
import math
import fdtd
import fdtd.backend as bd
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.optim as optim
import torchvision
from torch.utils.tensorboard import SummaryWriter
from autoencoder import AutoEncoder

#TODO - move this to a util file next cleanup
def get_sample_img(img_loader):
    _, (example_datas, labels) = next(enumerate(img_loader))
    sample = example_datas[0][0]
    sample = sample.to(device)[None, None, :]
    return sample

def get_object_by_name(grid, name):
    for obj in grid.objects:
        if(obj.name == name):
            return obj

# Tensorboard summary writer: outputs to ./runs/ directory
writer = SummaryWriter()

# ## Set Backend
backend_name = "torch"
#backend_name = "torch.cuda.float64"
fdtd.set_backend(backend_name)
if(backend_name.startswith("torch.cuda")):
    device = "cuda"
else:
    device = "cpu"

image_transform = torchvision.transforms.Compose([
                               torchvision.transforms.ToTensor()])
train_dataset = torchvision.datasets.CIFAR10('cifar10/', 
                                           train=True, 
                                           download=True,
                                           transform=image_transform)
train_loader = torch.utils.data.DataLoader(train_dataset,
                                           batch_size=1, 
                                           shuffle=True)


sample = get_sample_img(train_loader)
print('Image shape: ', sample.shape)
ih, iw = tuple(sample.shape[2:4])

# Physics constants
WAVELENGTH = 1550e-9
SPEED_LIGHT: float = 299_792_458.0  # [m/s] speed of light


# create FDTD Grid
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

grid[bw:bw+ih,bw:bw+iw,0] = fdtd.CorticalColumnPlaneSource(
    period = WAVELENGTH / SPEED_LIGHT,
    polarization = 'x', # BS value, polarization is not used.
    name='cc'
)

# grid[35,35,0] = fdtd.PointSource(
#     period = WAVELENGTH / SPEED_LIGHT,
#     name='ps'
# )

# Object defining the cortical column substrate 
grid[bw:-bw, bw:-bw, :] = fdtd.LearnableAnisotropicObject(permittivity=2.5, name="cc_substrate")


# Make the model
model = AutoEncoder(grid=grid, input_chans=1, output_chans=1).to(device)

print('All grid objects: ', [obj.name for obj in grid.objects])
params_to_learn = [get_object_by_name(grid, 'xlow').inverse_permittivity]
params_to_learn = [get_object_by_name(grid, 'xhigh').inverse_permittivity]
params_to_learn = [get_object_by_name(grid, 'ylow').inverse_permittivity]
params_to_learn = [get_object_by_name(grid, 'yhigh').inverse_permittivity]
params_to_learn = [get_object_by_name(grid, 'cc_substrate').inverse_permittivity]
params_to_learn += [*model.parameters()]

# Optimizer params
learning_rate = 0.01
optimizer = optim.SGD(params_to_learn, lr=learning_rate, momentum=0.5)
mse = torch.nn.MSELoss(reduce=False)
# Curriculum: loss emphasis (how much on em_loss)
alphas = [0.0, 0.01, 0.1, 0.5]
alpha_steps = [1000, 2000, 8000, math.inf]

max_train_steps = 1000000000000000
em_steps = 200 

grid.H.requires_grad = True
grid.H.retain_grad()
grid.E.requires_grad = True
grid.E.retain_grad()

loss_fn = torch.nn.MSELoss()

# Train the weights
counter = 0
print('Sum of perm: ', bd.sum(grid.objects[0].inverse_permittivity))
for train_step in range(max_train_steps):
    # Reset grid and optimizer
    grid.reset()
    optimizer.zero_grad()
    # Push it through Encoder
    if((train_step % 100 == 0) and (train_step > 0)):
        #vis = True
        vis = False
    else:
        vis = False
    # Get sample from training data
    img = get_sample_img(train_loader)
    img_hat_em, img_hat_aux, em_img = model(img, em_steps, visualize=vis)
    #img_hat_aux = model(img, em_steps, visualize=vis)
    # Add images to tensorboard
    img_grid = torchvision.utils.make_grid([img[0,...], img_hat_em, img_hat_aux[0,...], 
        torch.sum(em_img[0:3,...], axis=0, keepdim=True)])
    #img_grid = torchvision.utils.make_grid([img[0,...], img_hat_aux[0,...]])
    writer.add_image('images', img_grid, train_step)

    # Generate loss
    em_loss = loss_fn(img_hat_em, img) 
    aux_loss = loss_fn(img_hat_aux, img) 
    # Modify alpha based on training phase
    if(train_step > alpha_steps[0]):
        alphas.pop(0)
        alpha_steps.pop(0) 
    loss = alphas[0]*em_loss + (1-alphas[0])*aux_loss
    writer.add_scalar('EM Loss',  em_loss,  train_step)
    writer.add_scalar('Aux Loss', aux_loss, train_step)
    writer.add_scalar('Total Loss', loss, train_step)
    writer.add_scalar('ccsubstate_sum', 
            torch.sum(get_object_by_name(grid, 'cc_substrate').inverse_permittivity), train_step)
    print('Step: ', train_step, '\tTime: ', grid.time_steps_passed, '\tLoss: ', loss, '\tAlpha: ', alphas[0])

    # Tensorboard
    writer.add_histogram('cc_dirs', model.cc_dirs, train_step)
    writer.add_histogram('cc_freqs', model.cc_freqs, train_step)
    writer.add_histogram('cc_phases', model.cc_phases, train_step)
    writer.add_histogram('ccsubstrate', get_object_by_name(grid, 'cc_substrate').inverse_permittivity, train_step)
    writer.add_histogram('xlow', get_object_by_name(grid, 'xlow').inverse_permittivity, train_step)
    writer.add_histogram('xhigh', get_object_by_name(grid, 'xhigh').inverse_permittivity, train_step)
    writer.add_histogram('ylow', get_object_by_name(grid, 'ylow').inverse_permittivity, train_step)
    writer.add_histogram('yhigh', get_object_by_name(grid, 'yhigh').inverse_permittivity, train_step)

    optimizer.zero_grad()
    # Backprop
    loss.backward(retain_graph=True)
    optimizer.step()
    counter += 1
    # if(train_step % 20 == 0):
    #     img_to_show = torch.permute(y, (1,2,0)).detach().cpu()
    #     plt.imshow(img_to_show, cmap='gray', interpolation='none')
    #     plt.show()


writer.close()
