#!/usr/bin/env python

import sys
sys.path.append('/home/bij/Projects/fdtd/')
import fdtd
import fdtd.backend as bd
import matplotlib.pyplot as plt
import numpy as np


# ## Set Backend
#fdtd.set_backend("numpy")
fdtd.set_backend("torch")


# ## Simulation Constants
WAVELENGTH = 1550e-9 # For resolution purposes.
WAVELENGTH2 = 1550e-8 # For resolution purposes.
SPEED_LIGHT: float = 299_792_458.0  # [m/s] speed of light

# ## Simulation

# create FDTD Grid

# In[4]:


grid = fdtd.Grid(
    #(1.5e-5, 1.5e-5, 1),
    #(1.0e-5, 1.0e-5, 1), # Good ratios
    (1.5e-4, 1.5e-4, 1),
    #(2.5e-5, 2.5e-5, 1),
    grid_spacing=0.1 * WAVELENGTH,
    permittivity=1.0,
    permeability=1.0,
)
#grid.time_step = grid.time_step / 10

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

midpoint_y = grid.shape[0]//2
midpoint_x = grid.shape[1]//2
size = 300
# sources
for source_idx in range(10):
    y, x = (np.random.rand(2)*grid.shape[0]).astype(np.int)
    #y, x = (140, 100)
    y, x = (int(y), int(x))
    grid[ y, x, 0] = fdtd.PointSource(
        period=WAVELENGTH2 / SPEED_LIGHT, name="point_source_{0}".format(source_idx)
    )

grid[midpoint_y-size//2:midpoint_y+size//2, midpoint_x-size//2:midpoint_x+size//2, 0:1] = fdtd.AnisotropicObject(permittivity=54.0, name="object")

grid.visualize(z=0, animate=True, norm="log")
vis_steps = 100
step = 0
for i in range(1000000):
    grid.run(vis_steps, progress_bar=False)
    #grid.visualize(z=0, norm='log', animate=True)
    grid.visualize(z=0, norm='log', animate=True, srccolor=(0, 0, 0, 0), objedgecolor=(0,0,0,0), plot_both_fields=False, save=True, folder='./sim_frames/', index=i)
    plt.show()
    step += vis_steps
    print('On step: ', step)
x = input('type input to end: ')




