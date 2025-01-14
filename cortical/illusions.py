#!/usr/bin/env python
import git
import sys
import datetime
from pathlib import Path
sys.path.append('/home/bij/Projects/fdtd/')
import math
import time
import fdtd
import fdtd.backend as bd
import matplotlib.pyplot as plt
import numpy as np
import random
import torch
import torch.optim as optim
import torchvision
from torchvision.utils import save_image
from torch.utils.tensorboard import SummaryWriter
import scipy
from autoencoder import AutoEncoder
import argparse
from os import listdir
from os.path import isfile, join
from PIL import Image
import pickle

parser = argparse.ArgumentParser(description='Process args.')
parser.add_argument('-f', '--load-file', type=str, default=None,
                    help='File to load params from before training starts. Overrides --load-step.')
parser.add_argument('-c', '--coverage-ratio', type=float, default=1.0,
                    help='How much distance a wave can cover as a proportion of the diagonal length of the sim.')
parser.add_argument('-is', '--image-size', type=int, default=40,
                    help='Size of each side of the image. Determines grid size.')
parser.add_argument('-sc', '--image-scaler', type=int, default=1,
                    help='How much to scale the entire simulation by (changes the dimensions of the model).')
args = parser.parse_args()

def get_sorted_paths(directory_list, target_ext='.png'):
    path_list = []
    for directory in directory_list:
        paths = [join(directory,f) for f in listdir(directory) if isfile(join(directory, f)) and f.endswith(target_ext)]
        print(f'Found {len(paths)} files in {directory}')
        path_list += paths
    path_list.sort()
    return path_list

img_paths = get_sorted_paths(['./optical_illusions/'])

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
    else:
        print('Could not find object: ', name)
        sys.exit()
def get_source_by_name(grid, name):
    for src in grid.sources:
        if(src.name == name):
            return src 
    else:
        print('Could not find object: ', name)
        sys.exit()

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
    return normed_img 

class RandomRot90:
    # Randomly rotates the image by multiples of 90 degrees.
    def __init__(self):
        pass

    def __call__(self, sample):
        return torch.rot90(sample, k=random.randrange(4), dims=[1, 2])

# Setup model saving
model_parent_dir = './model_checkpoints/'
repo = git.Repo(search_parent_directories=True)
local_branch = repo.active_branch.name
model_checkpoint_dir = model_parent_dir + local_branch + '/'

# ## Set Backend
backend_name = "torch"
fdtd.set_backend(backend_name)
if(backend_name.startswith("torch.cuda")):
    device = "cuda"
else:
    device = "cpu"

image_transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Resize((args.image_size*args.image_scaler, args.image_size*args.image_scaler))])


ih, iw = (args.image_size*args.image_scaler, args.image_size*args.image_scaler)

# Physics constants
WAVELENGTH = 1550e-9 # meters
SPEED_LIGHT: float = 299_792_458.0  # [m/s] speed of light
GRID_SPACING = 0.1 * WAVELENGTH # meters

# Size of grid boundary layer
bw = 10*args.image_scaler
# Create FDTD Grid
grid_h, grid_w = (ih+bw*2, iw+bw*2)
print('Grid height and width: ', grid_h, grid_w)
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
# The number of steps is based on the coverage ratio.
em_steps = int(grid_diag_steps*args.coverage_ratio)


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
grid[bw:-bw, bw:-bw, :] = fdtd.LearnableAnisotropicObject(permittivity=2.5, is_substrate=True, name="cc_substrate")
# List all model checkpoints
checkpoints = [f for f in listdir(model_checkpoint_dir) if(isfile(join(model_checkpoint_dir, f)) and f.endswith('.pt'))]

torch.autograd.set_detect_anomaly(True)
# The weights for the reconstruction loss at each em time step. 
loss_step_weights = torch.ones(em_steps)/em_steps
#loss_step_weights = torch.nn.Parameter(torch.reshape(loss_step_weights, (-1, 1, 1, 1, 1)))
loss_step_weights.requires_grad = True
softmax = torch.nn.Softmax(dim=0)

# Initialize the model and grid with default params.
model = AutoEncoder(num_em_steps=em_steps, grid=grid, input_chans=3, output_chans=3).to(device)
print('All grid objects: ', [obj.name for obj in grid.objects])
grid_params_to_learn = []
grid_params_to_learn += [get_object_by_name(grid, 'xlow').inverse_permittivity]
grid_params_to_learn += [get_object_by_name(grid, 'xhigh').inverse_permittivity]
grid_params_to_learn += [get_object_by_name(grid, 'ylow').inverse_permittivity]
grid_params_to_learn += [get_object_by_name(grid, 'yhigh').inverse_permittivity]
grid_params_to_learn += [get_object_by_name(grid, 'cc_substrate').inverse_permittivity]
# Nonlinearity weights for the substrate. 
grid_params_to_learn += [get_object_by_name(grid, 'cc_substrate').nonlin_conv.weight]
grid_params_to_learn += [get_object_by_name(grid, 'cc_substrate').nonlin_conv.bias]
# Nonlinearity weights for the cortical columns. 
grid_params_to_learn += [get_source_by_name(grid, 'cc').nonlin_conv.weight]
grid_params_to_learn += [get_source_by_name(grid, 'cc').nonlin_conv.bias]
# The weights for the loss.
grid_params_to_learn += [loss_step_weights]
# Load saved params for model and optimizer.
checkpoint_steps = [int(cf.split('_')[-1].split('.')[0]) for cf in checkpoints]
if(args.load_file is not None):
    start_step = int(args.load_file.split('/')[-1].split('_')[-1].split('.')[0])
    print('Loading model {0}. Starting at step {1}.'.format(args.load_file, start_step))
    optimizer_path = args.load_file.rsplit('.', 1)[0] + '.opt'
    grid_path = args.load_file.rsplit('.', 1)[0] + '.grd'
    model.load_state_dict(torch.load(args.load_file))
else:
    if(args.load_step == 'latest'):
        if(len(checkpoint_steps) > 0):
            latest_idx = np.argmax(checkpoint_steps)
            start_step = checkpoint_steps[latest_idx]
            model_dict_path = model_checkpoint_dir + checkpoints[latest_idx]
            optimizer_path = model_dict_path.rsplit('.', 1)[0] + '.opt'
            grid_path = model_dict_path.rsplit('.', 1)[0] + '.grd'
            print('Loading model {0} with optimizer {1} and grid {2}.'.format(model_dict_path, optimizer_path, grid_path))
            model.load_state_dict(torch.load(model_dict_path))
        else:
            start_step = 0
    elif(int(args.load_step) != 0):
        if(int(args.load_step) not in checkpoint_steps):
            print('Checkpoint {0} not found in {1}'.format(args.load_step, model_checkpoint_dir))
            sys.exit()
        start_step = int(args.load_step)
        model_idx = np.where(np.array(checkpoint_steps) == start_step)[0][0]
        model_dict_path = model_checkpoint_dir + checkpoints[model_idx]
        optimizer_path = model_dict_path.rsplit('.', 1)[0] + '.opt'
        grid_path = model_dict_path.rsplit('.', 1)[0] + '.grd'
        print('Loading model {0} with optimizer {1} and grid {2}.'.format(model_dict_path, optimizer_path, grid_path))
        model.load_state_dict(torch.load(model_dict_path))
    else:
        print('Starting model at step 0')
        start_step = 0
        optimizer_path = None
        grid_path = None

def toy_img(img):
    img = torch.zeros_like(img)
    x, y, b, s = np.random.rand(4)
    max_size = 14
    min_size =  6
    max_b = 1.0
    min_b = 0.5
    x = int(x*(img.shape[-1] - max_size))
    y = int(y*(img.shape[-2] - max_size))
    b = float(min_b + b*(max_b - min_b))
    s = int(min_size + s*(max_size - min_size))
    img[..., x:x+s, y:y+s] = b
    return bd.array(img[:,0,...])

reset_optimizer = False
if((grid_path is not None)):
    print('Loading grid params...')
    with torch.no_grad():
        load_grid_params_to_learn = torch.load(grid_path)
        for idx, tensor in enumerate(load_grid_params_to_learn):
            #if(args.image_scaler == 1 or tensor.shape == grid_params_to_learn[idx][...].shape):
            if(tensor.shape == grid_params_to_learn[idx][...].shape):
                grid_params_to_learn[idx][...] = tensor[...]
            else:
                if(idx == len(load_grid_params_to_learn) - 1):
                    tensor = torch.squeeze(tensor)
                # Interpolate the thing....
                print('INFO: Shapes are mismatched: {0} vs {1}'.format(tensor[...].shape, grid_params_to_learn[idx][...].shape))
                
                # If this is a grid param, expand it over the spatial dims.
                if(len(tensor.shape) > 1):
                    reps = np.ones(len(tensor.shape), dtype=int)
                    for i in range(len(reps)):
                        reps[i] = args.image_scaler
                        if(i >= 1):
                            break

                    tensor_np_interp = scipy.ndimage.zoom(tensor.detach().numpy(), reps, order=1)
                    grid_params_to_learn[idx][...] = torch.from_numpy(tensor_np_interp)
                    print('INFO: Grid object scaled to shape: ', grid_params_to_learn[idx][...].shape)
                # If this is the loss step weights, scale it linearly to fit the new size.
                else:
                    tensor_interp = torch.nn.functional.interpolate(tensor[None, None, ...], grid_params_to_learn[idx][...].shape, mode='linear')
                    grid_params_to_learn[idx][...] = tensor_interp
                    print('INFO: EM Step loss object scaled to shape: ', grid_params_to_learn[idx][...].shape)
                    print('EM Steps: ', em_steps)
                    print('Loss step weights: ', loss_step_weights.shape)

                # Since parameter shapes have changed, the optimizer weights are obsolete.
                reset_optimizer = True

# Combine grid and model params and register them with the optimizer.

# Array of EM powers by illusion by time step
power_graph = {}
with torch.inference_mode():
    # Add images to tensorboard
    for img_idx, img_file in enumerate(img_paths):
        power_graph[img_file] = {'E': [], 'H': []}
        img = Image.open(img_file)
        img = image_transform(img)[None, ...]
        print(img_file, img.shape)
        # Reset grid
        grid.reset()
        for em_step, (img_hat_em, em_field) in enumerate(model(img)):
            print('Generating image for illusion {0} and em step {1}'.format(img_idx, em_step), end='\r')
            # Process outputs
            e_field_img = em_field[0:3,...]
            h_field_img = em_field[3:6,...]
            power_graph[img_file]['E'] += [torch.sum(e_field_img**2).numpy()]
            power_graph[img_file]['H'] += [torch.sum(h_field_img**2).numpy()]
            # Write to TB
            img_grid = torchvision.utils.make_grid([img[0,...], img_hat_em,
                norm_img_by_chan(e_field_img), 
                norm_img_by_chan(h_field_img)])
            img_grid = torchvision.transforms.functional.resize(img_grid, size=(img_grid.shape[1] * 4, img_grid.shape[2] * 4), interpolation=torchvision.transforms.InterpolationMode.NEAREST)

            save_image(img_grid, './images/img_p{0}_idx{1}.png'.format('{0:06.3f}'.format(img_idx), str(em_step).zfill(12)))


with open('./power_graph.pkl', 'wb') as f:
    pickle.dump(power_graph, f)
