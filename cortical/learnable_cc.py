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

grid[gl//2,gl//5,0] = fdtd.PointSource(
    period = WAVELENGTH / SPEED_LIGHT,
)

# detectors

# objects

midpoint_y = grid.shape[0]//2
midpoint_x = grid.shape[1]//2
grid[10:gl-10, midpoint_x-10:midpoint_x+10, 0:1] = fdtd.LearnableAnisotropicObject(permittivity=2.5, name="learnable_object")


# Add a detector
#my_detector = fdtd.LineDetector(name="detector")
#grid[midpoint_y-3:midpoint_y+3, midpoint_x+30, 0:1] = my_detector



momentum = 0.5
device = "cuda"
# Make the model
model = AutoEncoder().to(device)

print('Get object: ', [obj.name for obj in grid.objects])
#TODO - Add encoder trainable params to this list
#TODO - Add cc trainable params to this list
params_to_learn = [obj.inverse_permittivity for obj in grid.objects]
params_to_learn += [model.parameters()]
#learning_rate = 0.00001
#learning_rate = 1000
learning_rate = 0.01
optimizer = optim.SGD(params_to_learn, lr=learning_rate,
                      momentum=momentum)
mse = torch.nn.MSELoss(reduce=False)

max_train_steps = 100000
em_steps = 200 
visualizer_speed = 5

image_transform = torchvision.transforms.Compose([
                               torchvision.transforms.ToTensor()])
train_dataset = torchvision.datasets.CIFAR10('cifar10/', 
                                           train=True, 
                                           download=True,
                                           transform=image_transform)
#data loaders
train_loader = torch.utils.data.DataLoader(train_dataset,
                                           batch_size=batch_size_train, 
                                           shuffle=True)


def get_sample_img(img_loader):
    _, (example_datas, labels) = next(enumerate(img_loader))
    sample = example_datas[0][0]
    return sample.to(device)

sample = get_sample_img(train_loader)
# show the data
plt.imshow(sample, cmap='gray', interpolation='none')
print("Label: "+ str(labels[0]))

grid.H.requires_grad = True
grid.H.retain_grad()
grid.E.requires_grad = True
grid.E.retain_grad()

# Train the weights
counter = 0
print('Sum of perm: ', bd.sum(grid.objects[0].inverse_permittivity))
for train_step in range(max_train_steps):
    grid.reset()
    optimizer.zero_grad()
    ### X ### - Get a sample from training data
    img = get_sample_img(train_loader)
    ### X ### - Push it through Encoder
    Z = model(img)
    ### X ### - Seed CC with encoded stimulus
    ### X ### - Run sim

    # Run the simulator
    if(train_step % 10 == 0):
        for i in range(em_steps//visualizer_speed):
            grid.run(visualizer_speed, progress_bar=False)
            grid.visualize(z=0, norm='log', animate=True)
            plt.show()
    else:
        grid.run(em_steps , progress_bar=False)

    ### X ### - Decode EM field into an image
    ### X ### - Generate encoder loss
    detector_energy = bd.sum(bd.sum(grid.E[midpoint_y-3:midpoint_y+3, midpoint_x+30, 0:1] ** 2 
                            + grid.H[midpoint_y-3:midpoint_y+3, midpoint_x+30, 0:1] ** 2, -1))
    loss = -1.0*detector_energy
    print('Train step: ', train_step, '\tTime: ', grid.time_steps_passed, '\tLoss: ', loss, '\tDetector energy: ', detector_energy)
    optimizer.zero_grad()
    ### X ### - Backprop
    loss.backward(retain_graph=True)
    optimizer.step()
    counter += 1
    grid.visualize(z=0, norm='log', animate=True)
    plt.show()


