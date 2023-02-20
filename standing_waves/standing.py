import sys
sys.path.append('/home/bij/Projects/fdtd/')
import matplotlib.pyplot as plt

import fdtd
fdtd.set_backend("torch")

WAVELENGTH = 1550e-9
SPEED_LIGHT: float = 299_792_458.0  # [m/s] speed of light

# create FDTD Grid
grid = fdtd.Grid(
    #(1.5e-5, 1.5e-5, 1),  # 2D grid
    #(200, 100, 1),  # 2D grid
    (100, 40, 1),  # 2D grid
    grid_spacing=0.1 * WAVELENGTH,
    permittivity=2.5,  # same as object
)

# sources
grid[50, :] = fdtd.LineSource(period=WAVELENGTH / SPEED_LIGHT, name="source")

# x boundaries
# grid[0, :, :] = fdtd.PeriodicBoundary(name="xbounds")
#grid[0:10, :, :] = fdtd.PML(name="pml_xlow")
#grid[-10:, :, :] = fdtd.PML(name="pml_xhigh")

# y boundaries
# grid[:, 0, :] = fdtd.PeriodicBoundary(name="ybounds")
grid[:, 0:10, :] = fdtd.PML(name="pml_ylow")
grid[:, -10:, :] = fdtd.PML(name="pml_yhigh")

for i in range(1000):
    grid.run(1, progress_bar=False)
    grid.visualize(z=0, animate=True, norm="log")
