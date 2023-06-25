import numpy as np
import torch
import torchvision
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
#import cv2
import torchvision.transforms.functional as F
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm_notebook as tqdm
import util

plt.rcParams["savefig.bbox"] = 'tight'


## Then define the model class
class AutoEncoder(nn.Module):
    def __init__(self, grid, num_em_steps, input_chans=3, num_ccs=16, output_chans=3, wavelen_mean=1550e-3, freq_std_div=10):
        super(AutoEncoder, self).__init__()
        self.em_grid = grid
        self.num_em_steps = num_em_steps
        ic = input_chans
        cc = num_ccs
        oc = output_chans
        # Convolutions for common feature extractor
        self.conv1 = nn.Conv2d(ic,  8, kernel_size=5, stride=1, padding='same')
        self.conv2 = nn.Conv2d( 8, 16, kernel_size=5, stride=1, padding='same')
        self.conv3 = nn.Conv2d(16,  8, kernel_size=5, stride=1, padding='same')
        self.conv4 = nn.Conv2d( 8,  8, kernel_size=5, stride=1, padding='same')
        self.conv5 = nn.Conv2d( 8,  8, kernel_size=5, stride=1, padding='same')
        # Convs for CC activations
        self.cc_conv6 = nn.Conv2d( 8,  8, kernel_size=5, stride=1, padding='same')
        self.cc_conv7 = nn.Conv2d( 8, cc, kernel_size=5, stride=1, padding='same')
        # Convs for substrate manipulation
        self.sm_conv6 = nn.Conv2d( 8,  8, kernel_size=5, stride=1, padding='same')
        self.sm_conv7 = nn.Conv2d( 8,  9, kernel_size=5, stride=1, padding='same')
        # Converts E and H fields back into an image with a linear transformation
        self.conv_linear = nn.Conv2d(6, oc, kernel_size=1, stride=1, padding='same')
        # Converts cc_activations back into an image (for aux loss)
        self.conv_aux1 = nn.Conv2d( cc,  8, kernel_size=3, stride=1, padding='same')
        self.conv_aux2 = nn.Conv2d(  8,  8, kernel_size=3, stride=1, padding='same')
        self.conv_aux3 = nn.Conv2d(  8, oc, kernel_size=3, stride=1, padding='same')
        # Direction of E field perturbations
        # (output (E field), input (E field), kernel_T, kernel_H, kernel_W)
        # They must sum to zero and we just add them to the E field, no multiplication necessary
        #TODO - make sure these dir kernels make sense (check the sum)
        self.cc_dirs = torch.nn.Parameter(2*torch.rand((1, cc, 3, 3)) - 1)
        #TODO - remove this dumb line?
        self.cc_dirs = self.cc_dirs

        means = 1.0/wavelen_mean*torch.ones(num_ccs)
        stds = (means/freq_std_div)*torch.ones(num_ccs)
        self.cc_freqs  = torch.nn.Parameter(torch.normal(mean=means, std=stds))
        self.cc_phases = torch.nn.Parameter(torch.rand((num_ccs)))

    def get_em_plane(self):
        ' Extracts a slice along the image plane from the EM field. '
        em_plane = torch.cat([self.em_grid.E, self.em_grid.H], axis=-1)
        em_plane = em_plane[self.em_grid.sources[0].x, self.em_grid.sources[0].y]
        em_plane = torch.permute(torch.squeeze(em_plane), (2,0,1))
        return em_plane

    def forward(self, x, em_steps=None, amp_scaler=1.0):
        ## 1 - Extract features
        # Convert image into amplitude, frequency, and phase shift for our CCs.
        x = self.conv1(x)
        x = torch.relu(x)
        x = self.conv2(x)
        x = torch.relu(x)
        x = self.conv3(x)
        x = torch.relu(x)
        x = self.conv4(x)
        x = torch.relu(x)
        x = self.conv5(x)
        x = torch.relu(x)
        # Branch to cc activation
        x_cc = self.cc_conv6(x)
        x_cc = torch.relu(x_cc)
        x_cc = self.cc_conv7(x_cc)
        cc_activations = x_cc
        # Branch to substrate manipulation
        x_sm = self.sm_conv6(x)
        x_sm = torch.relu(x_sm)
        x_sm = self.sm_conv7(x_sm)
        sm_activations = x_sm

        ## 2 - Seed the cc grid source
        self.em_grid.sources[0].seed(cc_activations, self.cc_dirs, self.cc_freqs, self.cc_phases, amp_scaler)
        ## 3 - Seed the substrate
        util.get_object_by_name(self.em_grid, 'cc_substrate').seed(sm_activations)

        # 3 - Run the grid and generate output
        if(em_steps is None or em_steps == 0):
            em_steps = self.num_em_steps

        for em_step in range(em_steps):
            self.em_grid.run(1 , progress_bar=False)
            em_plane = self.get_em_plane()
            x_hat_em = torch.sigmoid(self.conv_linear(em_plane))
            yield x_hat_em, em_plane
