#!/usr/bin/env python

import sys
sys.path.append('/home/bij/PersonalProjects/fdtd/')
import fdtd
import fdtd.backend as bd
import matplotlib.pyplot as plt


# ## Set Backend
fdtd.set_backend("numpy")


# ## Constants
WAVELENGTH = 1550e-9
WAVELENGTH2 = 1550e-8
SPEED_LIGHT: float = 299_792_458.0  # [m/s] speed of light


# ## Simulation

# create FDTD Grid

# In[4]:


grid = fdtd.Grid(
    (2.5e-5, 1.5e-5, 1),
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
l1 = 1*grid.shape[0]//3
l2 = 2*grid.shape[0]//3

grid[l1, 70:75, 0] = fdtd.LineSource(
    period=WAVELENGTH / SPEED_LIGHT, name="linesource"
)
grid[l2, 70:75, 0] = fdtd.LineSource(
    period=WAVELENGTH / SPEED_LIGHT, name="linesource2",
)


# detectors

grid[12e-6, :, 0] = fdtd.LineDetector(name="detector")


# objects

#grid[11:32, 30:84, 0:1] = fdtd.NonLinearObject(permittivity=2.5, name="object")
width = 21
midline = grid.shape[0]//2
grid[midline-width//2:midline+width//2, 30:84, 0:1] = fdtd.NonLinearObject(permittivity=2.5, name="object")
#grid[11:32, 30:84, 0:1] = fdtd.AnisotropicObject(permittivity=2.5, name="object")


# ## Run simulation

# ## Visualization


grid.visualize(z=0, animate=True)
for i in range(1000):
    grid.run(1, progress_bar=False)
    grid.visualize(z=0, norm='log', animate=True)
    plt.show()
x = input('type input to end: ')



