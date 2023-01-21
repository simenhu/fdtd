#!/usr/bin/env python
# This is a test. We try to learn to represent a SINGLE image in the EM field.

import git
import sys
import datetime
sys.path.append('/home/bij/Projects/fdtd/')
import math
import time
import fdtd
import fdtd.backend as bd
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.optim as optim
import torchvision
from torch.utils.tensorboard import SummaryWriter
from autoencoder import DummyEncoder

model_checkpoint_dir = './model_checkpoints/'
#TODO - add functionality to bootstrap models
model_bootstrap_dir = './bootstrap/'

#TODO - move this to a util file next cleanup
def get_sample_img(img_loader, color=True):
    _, (example_datas, labels) = next(enumerate(img_loader))
    if(color):
        sample = example_datas[0]
        sample = sample.to(device)[None, :]
    else:
        sample = example_datas[0][0]
        sample = sample.to(device)[None, None, :]
    return sample

def get_object_by_name(grid, name):
    for obj in grid.objects:
        if(obj.name == name):
            return obj

def norm_img_by_chan(img):
    '''
    Puts each channel into the range [0,1].
    Expects input to be in CHW config.
    '''
    img_flat = torch.reshape(img, (3, -1))
    chan_maxes, _ = torch.max(img_flat, dim=-1, keepdims=True) 
    chan_mins, _  = torch.min(img_flat, dim=-1, keepdims=True) 
    chans_dynamic_range = chan_maxes - chan_mins
    normed_img = (img - chan_mins[...,None])/(chans_dynamic_range[...,None])
    #normed_img_flat = torch.reshape(normed_img, (3, -1))
    #print('Normed maxes: ', torch.max(normed_img_flat, dim=-1, keepdim=True)[0])
    #print('Normed mins: ', torch.min(normed_img_flat, dim=-1, keepdim=True)[0])
    return normed_img 




# Setup tensorboard
tb_parent_dir = './runs/'
repo = git.Repo(search_parent_directories=True)
sha = repo.head.object.hexsha
#head = repo.head
local_branch = repo.active_branch.name
log_dir = tb_parent_dir + local_branch + '/' + sha[-3:] + '/' +  datetime.datetime.now().isoformat(timespec='seconds') + '/'
print('TB Log Directory is: ', log_dir)
writer = SummaryWriter(log_dir=log_dir)

# ## Set Backend
backend_name = "torch"
#backend_name = "torch.cuda.float64"
fdtd.set_backend(backend_name)
if(backend_name.startswith("torch.cuda")):
    device = "cuda"
else:
    device = "cpu"

image_transform = torchvision.transforms.Compose([
                               torchvision.transforms.ToTensor()])
train_dataset = torchvision.datasets.CIFAR10('cifar10/', 
                                           train=True, 
                                           download=True,
                                           transform=image_transform)
#TODO - turn SHUFFLE back to TRUE for training on multiple images.
train_loader = torch.utils.data.DataLoader(train_dataset,
                                           batch_size=1, 
                                           shuffle=False)


sample = get_sample_img(train_loader)
print('Image shape: ', sample.shape)
ih, iw = tuple(sample.shape[2:4])

# Physics constants
WAVELENGTH = 1550e-9 # meters
SPEED_LIGHT: float = 299_792_458.0  # [m/s] speed of light
GRID_SPACING = 0.1 * WAVELENGTH # meters



# Size of grid boundary layer
bw = 10
# Create FDTD Grid
grid_h, grid_w = (ih+bw*2, iw+bw*2)
# Boundaries with width bw
grid = fdtd.Grid(
    (grid_h, grid_w, 1),
    grid_spacing=GRID_SPACING,
    permittivity=1.0,
    permeability=1.0,
)

# Calculate how long it takes a wave to cross the entire grid.
grid_diag_cells = math.sqrt(grid_h**2 + grid_w**2)
grid_diag_len = grid_diag_cells * GRID_SPACING
grid_diag_steps = int(grid_diag_len/SPEED_LIGHT/grid.time_step)+1
print('Time Steps to Cover Entire Grid: ', grid_diag_steps)


# Create learnable objects at the boundaries
grid[  0: bw, :, :] = fdtd.LearnableAnisotropicObject(permittivity=2.5, name="xlow")
grid[-bw:   , :, :] = fdtd.LearnableAnisotropicObject(permittivity=2.5, name="xhigh")
grid[:,   0:bw, :] = fdtd.LearnableAnisotropicObject(permittivity=2.5, name="ylow")
grid[:, -bw:  , :] = fdtd.LearnableAnisotropicObject(permittivity=2.5, name="yhigh")
grid[:, :, 0] = fdtd.PeriodicBoundary(name="zbounds")



# Creat the cortical column sources
grid[bw:bw+ih,bw:bw+iw,0] = fdtd.CorticalColumnPlaneSource(
    period = WAVELENGTH / SPEED_LIGHT,
    polarization = 'x', # BS value, polarization is not used.
    name='cc'
)

# Object defining the cortical column substrate 
grid[bw:-bw, bw:-bw, :] = fdtd.LearnableAnisotropicObject(permittivity=2.5, name="cc_substrate")


# We only ever get ONE image to train on
img = get_sample_img(train_loader)

# Make the model
dummy_model = DummyEncoder(grid=grid, input_img=img, input_chans=3, output_chans=3).to(device)

print('All grid objects: ', [obj.name for obj in grid.objects])
params_to_learn = [get_object_by_name(grid, 'xlow').inverse_permittivity]
params_to_learn = [get_object_by_name(grid, 'xhigh').inverse_permittivity]
params_to_learn = [get_object_by_name(grid, 'ylow').inverse_permittivity]
params_to_learn = [get_object_by_name(grid, 'yhigh').inverse_permittivity]
#TODO - disabling substrate learning for now
#params_to_learn = [get_object_by_name(grid, 'cc_substrate').inverse_permittivity]
params_to_learn += [*dummy_model.parameters()]

# Optimizer params
learning_rate = 0.01
optimizer = optim.SGD(params_to_learn, lr=learning_rate, momentum=0.5)
mse = torch.nn.MSELoss(reduce=False)
loss_fn = torch.nn.MSELoss()

max_train_steps = 1000000000000000
save_interval = 1000
em_steps = 200 

grid.H.requires_grad = True
grid.H.retain_grad()
grid.E.requires_grad = True
grid.E.retain_grad()

# For timing steps
stopwatch = time.time()

# Train the weights
for train_step in range(max_train_steps):
    # Reset grid and optimizer
    grid.reset()
    optimizer.zero_grad()
    # Push it through Encoder
    #if((train_step % 100 == 0) and (train_step > 0)):
    if((train_step % 500 == 0)):
        vis = True
    else:
        vis = False

    num_samples = 2 
    # Get sample from training data
    img_hat_em, em_field = dummy_model(img, min_em_steps=grid_diag_steps, max_em_steps=2*grid_diag_steps, num_samples=num_samples, visualize=vis)
    e_field_img = em_field[:, 0:3,...]
    h_field_img = em_field[:, 3:6,...]

    # Add images to tensorboard
    for s in range(num_samples):
        img_grid = torchvision.utils.make_grid([img[0,...], img_hat_em[s],
            norm_img_by_chan(e_field_img[s]), 
            norm_img_by_chan(h_field_img[s])])
        writer.add_image('sample_'+str(s), img_grid, train_step)

    perm = torch.reshape(get_object_by_name(grid, 'cc_substrate').inverse_permittivity, (-1, 32, 32))
    writer.add_image('ccsubstrate1', perm[0:3,...], train_step)
    writer.add_image('ccsubstrate2', perm[3:6,...], train_step)
    writer.add_image('ccsubstrate3', perm[6:9,...], train_step)

    # Generate loss
    loss = loss_fn(img_hat_em, img) 

    writer.add_scalar('Total Loss', loss, train_step)
    writer.add_scalar('ccsubstate_sum', 
            torch.sum(get_object_by_name(grid, 'cc_substrate').inverse_permittivity), train_step)

    print('Step: ', train_step, '\tTime: ', grid.time_steps_passed, '\tLoss: ', loss)

    # Tensorboard
    writer.add_histogram('cc_dirs', dummy_model.cc_dirs, train_step)
    writer.add_histogram('cc_freqs', dummy_model.cc_freqs, train_step)
    writer.add_histogram('cc_phases', dummy_model.cc_phases, train_step)
    writer.add_histogram('ccsubstrate', get_object_by_name(grid, 'cc_substrate').inverse_permittivity, train_step)
    writer.add_histogram('xlow', get_object_by_name(grid, 'xlow').inverse_permittivity, train_step)
    writer.add_histogram('xhigh', get_object_by_name(grid, 'xhigh').inverse_permittivity, train_step)
    writer.add_histogram('ylow', get_object_by_name(grid, 'ylow').inverse_permittivity, train_step)
    writer.add_histogram('yhigh', get_object_by_name(grid, 'yhigh').inverse_permittivity, train_step)
    writer.add_histogram('e_field', e_field_img, train_step)
    writer.add_histogram('h_field', h_field_img, train_step)

    optimizer.zero_grad()
    # Backprop
    loss.backward(retain_graph=True)
    optimizer.step()

    # Save model 
    if((train_step % save_interval == 0) and (train_step > 0)):
        torch.save(dummy_model.state_dict(), model_checkpoint_dir + 'model_' + str(train_step).zfill(12) + '.pt')

    # Profile performance
    seconds_per_step = time.time() - stopwatch 
    writer.add_scalar('seconds_per_step', torch.tensor(seconds_per_step), train_step)
    stopwatch = time.time()

writer.close()
