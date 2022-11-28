#!/usr/bin/env python

import sys
sys.path.append('/home/bij/PersonalProjects/fdtd/')
import fdtd
import fdtd.backend as bd
import matplotlib.pyplot as plt


# Set Backend
fdtd.set_backend("torch")


# ## Constants
WAVELENGTH = 1550e-9
WAVELENGTH2 = 1550e-8
SPEED_LIGHT: float = 299_792_458.0  # [m/s] speed of light


class EMSimulator():
    def __init__(self):

        # create FDTD Grid
        #TODO - make the permittivity and permeability learnable.
        self.grid = fdtd.Grid(
            (1.5e-5, 1.5e-5, 1),
            grid_spacing=0.1 * WAVELENGTH,
            permittivity=1.0,
            permeability=1.0,
        )

        print('Grid Shape: ', self.grid.shape)


        # boundaries

        self.grid[ 0:10,    :,   :] = fdtd.PML(name="pml_xlow")
        self.grid[ -10:,    :,   :] = fdtd.PML(name="pml_xhigh")
        self.grid[    :, 0:10,   :] = fdtd.PML(name="pml_ylow")
        self.grid[    :, -10:,   :] = fdtd.PML(name="pml_yhigh")

        self.grid[:, :, 0] = fdtd.PeriodicBoundary(name="zbounds")


        # Sources - TODO - these should match every pixel and should be overwritable.
        self.grid[20, -20, 0] = fdtd.CorticalColumnPointSource(dir_vec=(0,1,0),
            period=WAVELENGTH / SPEED_LIGHT, name="linesource0"
        )
        self.grid[-20, 20, 0] = fdtd.CorticalColumnPointSource(dir_vec=(1,0,0),
            period=WAVELENGTH / SPEED_LIGHT, name="linesource3",
        )


        # Objects - TODO - these should have learnable params.

        midpoint_y = self.grid.shape[0]//2
        midpoint_x = self.grid.shape[1]//2
        self.grid[midpoint_y-10:midpoint_y+10, midpoint_x-10:midpoint_x+10, 0:1] = fdtd.AnisotropicObject(permittivity=2.5, name="object")


    # ## Run simulation and Visualization
    def run_sim(self, steps, visualize=False):
        ''' Runs the simulation and returns the E and H fields '''
        if(visualize):
            self.grid.visualize(z=0, animate=True)
        self.grid.run(steps, progress_bar=False)
        if(visualize):
            self.grid.visualize(z=0, norm='log', animate=True)
            plt.show()

        return self.grid.E, self.grid.H

