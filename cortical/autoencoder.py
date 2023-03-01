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


plt.rcParams["savefig.bbox"] = 'tight'

def img_norm(img):
    '''
    Normalizes an image's dynamic range to the interval (0,1)
    '''
    return (img - np.min(img)) / (np.max(img) - np.min(img))

def format_imgs(data, output, rows=3):
    '''
    Assumes shape of (bs, chans, H, W)
    '''
    # Convert to numpy
    data = data.detach().cpu().numpy()
    output = output.detach().cpu().numpy()
    # Normalize 
    data = img_norm(data)
    output = img_norm(output)
    # Make an output of shape (rows*H, 2*W, 3)
    img = np.zeros((rows*data.shape[2], 2*data.shape[3], 3))
    for row in range(rows):
        # Get the input img
        img[row*data.shape[2]:(row+1)*data.shape[2], 0:data.shape[3], :] = np.transpose(data[row], (1,2,0))
        # Get the output img
        img[row*data.shape[2]:(row+1)*data.shape[2], data.shape[3]:2*data.shape[3], :] = np.transpose(output[row], (1,2,0))

    return img

## Then define the model class
class AutoEncoder(nn.Module):
    def __init__(self, grid, input_chans=3, num_ccs=16, output_chans=3, wavelen_mean=1550e-3, freq_std_div=10):
        super(AutoEncoder, self).__init__()
        self.em_grid = grid
        ic = input_chans
        cc = num_ccs
        oc = output_chans
        self.conv1 = nn.Conv2d(ic,  8, kernel_size=5, stride=1, padding='same')
        self.conv2 = nn.Conv2d( 8, 16, kernel_size=5, stride=1, padding='same')
        self.conv3 = nn.Conv2d(16,  8, kernel_size=5, stride=1, padding='same')
        self.conv4 = nn.Conv2d( 8,  8, kernel_size=5, stride=1, padding='same')
        self.conv5 = nn.Conv2d( 8,  8, kernel_size=5, stride=1, padding='same')
        self.conv6 = nn.Conv2d( 8,  8, kernel_size=5, stride=1, padding='same')
        self.conv7 = nn.Conv2d( 8, cc, kernel_size=5, stride=1, padding='same')
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

    def forward(self, x, em_steps, visualize=False, visualizer_speed=5):
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
        x = self.conv6(x)
        x = torch.relu(x)
        x = self.conv7(x)
        cc_activations = x

        ## 2 - Seed the cc grid source
        self.em_grid.sources[0].seed(cc_activations, self.cc_dirs, self.cc_freqs, self.cc_phases)

        # 3 - Run the grid and generate output
        if(visualize):
            num_iters = em_steps//visualizer_speed + 1
            rem_steps = em_steps %visualizer_speed
            for i in range(num_iters):
                if(i < num_iters-1):
                    self.em_grid.run(visualizer_speed, progress_bar=False)
                else:
                    self.em_grid.run(rem_steps, progress_bar=False)
                em_plane = self.get_em_plane()
                x_hat_em = torch.sigmoid(self.conv_linear(em_plane))
                yield x_hat_em, em_plane
        else:
            self.em_grid.run(em_steps , progress_bar=False)
            em_plane = self.get_em_plane()
            x_hat_em = torch.sigmoid(self.conv_linear(em_plane))
            yield x_hat_em, em_plane
