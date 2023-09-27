#!/usr/bin/env python

import sys
sys.path.append('/home/bij/Projects/fdtd/')
import fdtd
import fdtd.backend as bd
import matplotlib.pyplot as plt
from PIL import Image
import torch
import numpy as np



# ## Set Backend
#fdtd.set_backend("numpy")
fdtd.set_backend("torch")


# ## Simulation Constants
WAVELENGTH = 1550e-9 # For resolution purposes.
SPEED_LIGHT: float = 299_792_458.0  # [m/s] speed of light

# Frequency of the brainwaves.
#FREQ = 34000000000000.0 # Hz
#FREQ = 3400000000000000.0 # Hz
#FREQ = 3400000000.0 # Hz
FREQ = 340000000000.0 # Hz


# ## Simulation

# create FDTD Grid

# In[4]:


grid = fdtd.Grid(
    (2.5e-5, 2.5e-5, 1),
    grid_spacing=0.1 * WAVELENGTH,
    permittivity=1.0,
    permeability=1.0,
)

print('Grid Shape: ', grid.shape)


# boundaries

# In[5]:


# grid[0, :, :] = fdtd.PeriodicBoundary(name="xbounds")
grid[0:10, :, :] = fdtd.PML(name="pml_xlow")
grid[-10:, :, :] = fdtd.PML(name="pml_xhigh")

# grid[:, 0, :] = fdtd.PeriodicBoundary(name="ybounds")
grid[:, 0:10, :] = fdtd.PML(name="pml_ylow")
grid[:, -10:, :] = fdtd.PML(name="pml_yhigh")

grid[:, :, 0] = fdtd.PeriodicBoundary(name="zbounds")


# sources
xe = grid.shape[0] - 10
ye = grid.shape[1] - 10

grid[ 10, 10:xe, 0] = fdtd.LineSource(
    period=1.0 / FREQ, name="linesource0"
)
#grid[ye-1, 10:xe, 0] = fdtd.LineSource(
#    period=1.0 / FREQ, name="linesource1"
#)
#grid[ 10:ye, 10, 0] = fdtd.LineSource(
#    period=1.0 / FREQ, name="linesource2"
#)
#grid[10:ye, xe-1, 0] = fdtd.LineSource(
#    period=1.0 / FREQ, name="linesource3"
#)


# detectors

# grid[12e-6, :, 0] = fdtd.LineDetector(name="detector")


# objects

midpoint_y = grid.shape[0]//2
midpoint_x = grid.shape[1]//2
size = 20

#grid[midpoint_x:-xe, :, 0] = fdtd.Object(permittivity=100, name="object")
grid[midpoint_x:xe, :, 0] = fdtd.Object(permittivity=10000, name="object")


# ## Run simulation

# ## Visualization


grid.visualize(z=0, animate=True, norm="log")
vis_steps = 1000
step = 0
for i in range(1000000):
    grid.run(vis_steps, progress_bar=False)
    grid.visualize(z=0, norm='log', animate=True)
    #grid.visualize(z=0, norm='log', animate=True, objcolor=(0, 0, 0, 0), objedgecolor=(1,1,1,1), plot_both_fields=False, save=True, folder='./sim_frames/', index=i)
    plt.show()
    step += vis_steps
    print('On step: ', step)
x = input('type input to end: ')




