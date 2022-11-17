#!/usr/bin/env python

import sys
sys.path.append('/home/bij/PersonalProjects/fdtd/')
import fdtd
import fdtd.backend as bd
import matplotlib.pyplot as plt


# ## Set Backend
#fdtd.set_backend("numpy")
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

# grid[50, 70:75, 0] = fdtd.LineSource(
#     period=WAVELENGTH / SPEED_LIGHT, name="linesource"
# )
# grid[70, 70:75, 0] = fdtd.LineSource(
#     period=WAVELENGTH / SPEED_LIGHT, name="linesource2",
# )

grid[20, -20, 0] = fdtd.CorticalColumnPointSource(dir_vec=(0,1,0),
    period=WAVELENGTH / SPEED_LIGHT, name="linesource0"
)
grid[-20, 20, 0] = fdtd.CorticalColumnPointSource(dir_vec=(1,0,0),
    period=WAVELENGTH / SPEED_LIGHT, name="linesource3",
)


# detectors

# grid[12e-6, :, 0] = fdtd.LineDetector(name="detector")


# objects

midpoint_y = grid.shape[0]//2
midpoint_x = grid.shape[1]//2
grid[midpoint_y-10:midpoint_y+10, midpoint_x-10:midpoint_x+10, 0:1] = fdtd.AnisotropicObject(permittivity=2.5, name="object")


# ## Run simulation

# ## Visualization


grid.visualize(z=0, animate=True)
#for i in range(1000):
#    grid.run(1, progress_bar=False)
grid.run(100, progress_bar=False)
grid.visualize(z=0, norm='log', animate=True)
plt.show()
x = input('type input to end: ')




