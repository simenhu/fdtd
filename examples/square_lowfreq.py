#!/usr/bin/env python

import sys
sys.path.append('/home/bij/Projects/fdtd/')
import fdtd
import fdtd.backend as bd
import matplotlib.pyplot as plt


# ## Set Backend
#fdtd.set_backend("numpy")
fdtd.set_backend("torch")


# ## Simulation Constants
WAVELENGTH = 1550e-9 # For resolution purposes.
SPEED_LIGHT: float = 299_792_458.0  # [m/s] speed of light

# Frequency of the brainwaves.
BRAIN_FREQ = 34.0 # Hz

input('lkjwelfjk')


# ## Simulation

# create FDTD Grid

# In[4]:


grid = fdtd.Grid(
    #(1.5e-5, 1.5e-5, 1),
    #(1.0e-5, 1.0e-5, 1), # Good ratios
    #(1.5e-4, 1.5e-4, 1),
    (2.5e-5, 2.5e-5, 1),
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
xe = grid.shape[0] - 10
ye = grid.shape[1] - 10

grid[ 10, 10:xe, 0] = fdtd.LineSource(
    period=1.0 / BRAIN_FREQ, name="linesource0"
)
#grid[ye-1, 10:xe, 0] = fdtd.LineSource(
#    period=1.0 / BRAIN_FREQ, name="linesource1"
#)
#grid[ 10:ye, 10, 0] = fdtd.LineSource(
#    period=1.0 / BRAIN_FREQ, name="linesource2"
#)
#grid[10:ye, xe-1, 0] = fdtd.LineSource(
#    period=1.0 / BRAIN_FREQ, name="linesource3"
#)


# detectors

# grid[12e-6, :, 0] = fdtd.LineDetector(name="detector")


# objects

midpoint_y = grid.shape[0]//2
midpoint_x = grid.shape[1]//2
size = 20
#grid[midpoint_y-40:midpoint_y+40, midpoint_x-40:midpoint_x+40, 0:1] = fdtd.AnisotropicObject(permittivity=250000000000000000.0, name="object")
#grid[midpoint_y-size//2:midpoint_y+size//2, midpoint_x-size//2:midpoint_x+size//2, 0:1] = fdtd.AnisotropicObject(permittivity=250000000000000000.0, name="object")
#grid[midpoint_y-size//2:midpoint_y+size//2, midpoint_x-size//2:midpoint_x+size//2, 0:1] = fdtd.AnisotropicObject(permittivity=2500000.0, name="object")
#grid[midpoint_y-size//2:midpoint_y+size//2, midpoint_x-size//2:midpoint_x+size//2, 0:1] = fdtd.AnisotropicObject(permittivity=54.0, name="object")
grid[midpoint_y-size//2:midpoint_y+size//2, midpoint_x-size//2:midpoint_x+size//2, 0:1] = fdtd.AnisotropicObject(permittivity=54000.0, name="object")

# grid[midpoint_y, midpoint_x, 0] = fdtd.PointSource(
#     period=WAVELENGTH2 / SPEED_LIGHT, amplitude=0.001, name="pointsource0"
# )

# ## Run simulation

# ## Visualization


grid.visualize(z=0, animate=True, norm="log")
vis_steps = 100
step = 0
for i in range(1000000):
    grid.run(vis_steps, progress_bar=False)
    #grid.visualize(z=0, norm='log', animate=True)
    #grid.visualize(z=0, norm='log', animate=True, objcolor=(0, 0, 0, 0), objedgecolor=(1,1,1,1), plot_both_fields=False, save=True, folder='./sim_frames/', index=i)
    grid.visualize(z=0, norm='log', animate=True, objcolor=(0, 0, 0, 0), objedgecolor=(0,0,0,0), plot_both_fields=False, save=True, folder='./sim_frames/', index=i)
    plt.show()
    step += vis_steps
    print('On step: ', step)
x = input('type input to end: ')




