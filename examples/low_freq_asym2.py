#!/usr/bin/env python

import sys
sys.path.append('/home/bij/Projects/fdtd/')
import fdtd
import fdtd.backend as bd
import matplotlib.pyplot as plt


# ## Set Backend
#fdtd.set_backend("numpy")
fdtd.set_backend("torch")


# ## Constants
WAVELENGTH = 1550e-9
#WAVELENGTH2 = 1550e-7
#WAVELENGTH2 = 1550e-5
#WAVELENGTH2 = 1550e-6
WAVELENGTH2 = 1550e-2
SPEED_LIGHT: float = 299_792_458.0  # [m/s] speed of light


# ## Simulation

# create FDTD Grid

# In[4]:


grid = fdtd.Grid(
    #(1.5e-5, 1.5e-5, 1),
    #(1.5e-4, 1.5e-4, 1),
    (10.5e-6, 10.5e-6, 1),
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


# sources

# grid[50, 70:75, 0] = fdtd.LineSource(
#     period=WAVELENGTH / SPEED_LIGHT, name="linesource"
# )
# grid[70, 70:75, 0] = fdtd.LineSource(
#     period=WAVELENGTH / SPEED_LIGHT, name="linesource2",
# )

#grid[20, -20:-25, 0] = fdtd.LineSource(
#    period=WAVELENGTH2 / SPEED_LIGHT, name="linesource0"
#)
#grid[-20, -20:-25, 0] = fdtd.LineSource(
#    period=WAVELENGTH2 / SPEED_LIGHT, name="linesource1"
#)
#grid[20, 20:25, 0] = fdtd.LineSource(
#    period=WAVELENGTH2 / SPEED_LIGHT, name="linesource2",
#)
#grid[-20, 20:25, 0] = fdtd.LineSource(
#    period=WAVELENGTH2 / SPEED_LIGHT, name="linesource3",
#)


# detectors

# grid[12e-6, :, 0] = fdtd.LineDetector(name="detector")


# objects

midpoint_y = grid.shape[0]//2
midpoint_x = grid.shape[1]//2
#grid[midpoint_y-40:midpoint_y+40, midpoint_x-40:midpoint_x+40, 0:1] = fdtd.AnisotropicObject(permittivity=250000000000000000.0, name="object")
#grid[midpoint_y-10:midpoint_y+10, midpoint_x-10:midpoint_x+10, 0:1] = fdtd.AnisotropicObject(permittivity=250000000000000000.0, name="object")
#
grid[midpoint_y-20:midpoint_y+20, midpoint_x-20:midpoint_x+20, 0] = fdtd.AnisotropicObject(permittivity=250.0, name="object")
grid[midpoint_y-4:midpoint_y+4, midpoint_x-4:midpoint_x+4, 0:1] = fdtd.PlaneSource(
    period=WAVELENGTH2 / SPEED_LIGHT, name="linesource4", polarization = 'x'
)
# ## Run simulation

# ## Visualization


grid.visualize(z=0, animate=True)
for i in range(1000000):
    grid.run(1, progress_bar=False)
    grid.visualize(z=0, norm='log', animate=True)
    plt.show()
x = input('type input to end: ')




