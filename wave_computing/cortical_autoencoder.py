import numpy as np
import torch
import torchvision
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
import torchvision.transforms.functional as F
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm_notebook as tqdm
from EMSim import EMSimulator
from torch.autograd import Variable

class MaxwellLayer:
    def __init__(self, num_filters, h, w):
        # Generate weights that add up to 1.
        self.weights = Variable(torch.randn(1, num_filters, h, w))
        self.weights = self.weights/torch.sum(self.weights)
        #TODO - get a decent starting frequency
        self.freq = np.pi/2.0
        self.num_filters = num_filters
        print('weights: ', torch.sum(self.weights), self.weights)
        
    def call(self, x, t):        
        #print('x:', x.shape)
        x_ones = torch.ones_like(x)
        neg_divergence = torch.nn.functional.conv2d(x_ones, self.weights, padding='valid', groups=self.num_filters)
        neg_divergence = torch.nn.functional.pad(neg_divergence, pad=(1,1,1,1))
        #print('nd', neg_divergence.shape)
        pos_divergence = output_ct = torch.nn.functional.conv_transpose2d(x_ones[...,1:-1,1:-1], self.weights, groups=self.num_filters)
        #print('pd', pos_divergence.shape)
        osc = np.sin(self.freq*t)
        output = x + osc*(neg_divergence - pos_divergence)
        output = output
        #print(output.shape)
        return output


## Then define the model class
class WaveAutoEncoder(nn.Module):
    def __init__(self, input_chans=3, em_chans=3, load_weights=False):
        super(WaveAutoEncoder, self).__init__()

        # The EM simulator 
        self.emsim = EMSimulator()

        ic = input_chans
        em = em_chans

        # Activation functions
        self.fex_act = torch.relu
        self.dec_act = torch.relu

        # Feature extraction layers.
        self.fex_conv1 = nn.Conv2d(ic, 16, kernel_size=5, stride=1, padding='same')
        self.fex_conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=1, padding='same')
        self.fex_conv3 = nn.Conv2d(32, 16, kernel_size=5, stride=1, padding='same')
        self.fex_conv4 = nn.Conv2d(16, em, kernel_size=5, stride=1, padding='same')

        # Maxwell layer.
        self.mw = MaxwellLayer()

        # EM Decoder layers.
        #TODO - find out how many layers the EM field makes up and replace it below.
        self.dec_conv1 = nn.Conv2d(em, 16, kernel_size=5, stride=1, padding='same')
        self.dec_conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=1, padding='same')
        self.dec_conv3 = nn.Conv2d(32, 16, kernel_size=5, stride=1, padding='same')
        self.dec_conv4 = nn.Conv2d(16, ic, kernel_size=5, stride=1, padding='same')

        # The rise intensity of the signal (how fast it fires).
        self.frequencies = (oc)
        # The direction of the charge when fired (vector in 3D).
        self.directions = (oc, 3)

        #TODO - decide what this looks like.
        # Malleability (how much the surrounding fields affect this neuron):
        # We have to find a relationship b/t firing strength and E-field source behavior.

        # A source should shake a charge back and forth in the direction of self.direction
        # and the magnitude decides how much of a charge to shake.

        
    def forward(self, x):
        x = self.fex_conv1(x)
        x = torch.relu(x)
        x = self.fex_conv2(x)
        x = torch.relu(x)
        x = self.fex_conv3(x)
        x = torch.relu(x)
        x = self.fex_conv4(x)
        # Correct activation function?
        x = torch.relu(x)

        # Convert activations to EM perturbation magnitudes with current E or H fields as inputs.
        mags = f(E_field, x)

        # Setup the simulation w/ the source mags
        self.emsim.set_activations(mags, self.frequencies, self.directions)

        # Run the EM simulation for N steps and get the resulting E and H fields
        E, H = self.emsim.run(self.num_EM_steps)

        # Combine the E and H fields
        x = bd.stack([E,H])

        # Feed E and H to decoder
        x = self.dec_conv1(x)
        x = torch.relu(x)
        x = self.dec_conv2(x)
        x = torch.relu(x)
        x = self.dec_conv3(x)
        x = torch.relu(x)
        x = self.dec_conv4(x)
        # Correct activation function?
        x = torch.sigmoid(x)
        return x

wae = WaveAutoEncoder()
img = torch.zeros(1,3,200,200)
img_hat = wae(img)
