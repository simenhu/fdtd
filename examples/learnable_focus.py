#!/usr/bin/env python

# Here we demonstrate learning parameters.

import sys
sys.path.append('/home/bij/Projects/fdtd/')
import fdtd
import fdtd.backend as bd
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.optim as optim


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

han_pulse_len = 100000000000000000
pulse = False

print('Grid Shape: ', grid.shape)


# boundaries

# In[5]:


# For some reason these don't reset properly.
# # grid[0, :, :] = fdtd.PeriodicBoundary(name="xbounds")
# grid[0:10, :, :] = fdtd.PML(name="pml_xlow")
# grid[-10:, :, :] = fdtd.PML(name="pml_xhigh")
# 
# # grid[:, 0, :] = fdtd.PeriodicBoundary(name="ybounds")
# grid[:, 0:10, :] = fdtd.PML(name="pml_ylow")
# grid[:, -10:, :] = fdtd.PML(name="pml_yhigh")

grid[:, :, 0] = fdtd.PeriodicBoundary(name="zbounds")


# sources

gl = grid.shape[0]

grid[gl//2,gl//5,0] = fdtd.PointSource(
    period = WAVELENGTH / SPEED_LIGHT,
    pulse = pulse,
    cycle = han_pulse_len,
)

# detectors

# objects

midpoint_y = grid.shape[0]//2
midpoint_x = grid.shape[1]//2
grid[10:gl-10, midpoint_x-10:midpoint_x+10, 0:1] = fdtd.LearnableAnisotropicObject(permittivity=2.5, name="learnable_object")


# Add a detector
#my_detector = fdtd.LineDetector(name="detector")
#grid[midpoint_y-3:midpoint_y+3, midpoint_x+30, 0:1] = my_detector


#learning_rate = 0.00001
#learning_rate = 1000
learning_rate = 0.01
momentum = 0.5
device = "cuda"
print('Get object: ', grid.objects[0].name)
optimizer = optim.SGD([grid.objects[0].inverse_permittivity], lr=learning_rate,
                      momentum=momentum)
mse = torch.nn.MSELoss(reduce=False)

max_train_steps = 100000
em_steps = 200 
#grid.visualize(z=0, animate=True)
#torch.autograd.set_detect_anomaly(True)

grid.H.requires_grad = True
grid.H.retain_grad()
grid.E.requires_grad = True
grid.E.retain_grad()

# Train the weights
counter = 0
print('Sum of perm: ', bd.sum(grid.objects[0].inverse_permittivity))
for train_step in range(max_train_steps):
    grid.reset()
    grid.objects[0].inverse_permittivity.detach()
    optimizer.zero_grad()
    grid.E.detach()
    grid.H.detach()
    # Reset the grid
    grid.run(em_steps , progress_bar=False)
    print('Train step: ', train_step, '\tTime: ', grid.time_steps_passed)
    detector_energy = bd.sum(bd.sum(grid.E[midpoint_y-3:midpoint_y+3, midpoint_x+30, 0:1] ** 2 
                            + grid.H[midpoint_y-3:midpoint_y+3, midpoint_x+30, 0:1] ** 2, -1))
    loss = -1.0*detector_energy
    print('Loss: ', loss, '\tDetector energy: ', detector_energy)
    optimizer.zero_grad()
    loss.backward(retain_graph=True)
    optimizer.step()
    counter += 1
    grid.visualize(z=0, norm='log', animate=True)
    plt.show()
    #print('Sum of perm: ', bd.sum(grid.objects[0].inverse_permittivity))
    #print('Sum of E after: ', bd.sum(grid.E))

