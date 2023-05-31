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
from torch.utils.tensorboard import SummaryWriter
from autoencoder import AutoEncoder
import argparse
from os import listdir
from os.path import isfile, join

parser = argparse.ArgumentParser(description='Process args.')
parser.add_argument('-f', '--load-file', type=str, default=None,
                    help='File to load params from before training starts. Overrides --load-step.')
parser.add_argument('-l', '--load-step', type=str, default='0',
                    help='Where to start training. If latest, will start at the latest checkpoint.')
parser.add_argument('-s', '--save-steps', type=int, default='1000',
                    help='How often to save the model.')
parser.add_argument('-c', '--coverage-ratio', type=float, default=1.0,
                    help='How much distance a wave can cover as a proportion of the diagonal length of the sim.')
parser.add_argument('-m', '--max-steps', type=int, default='1000000000000000',
                    help='How many steps to train.')
parser.add_argument('-d', '--dry-run', type=bool, default=False,
                    help='If true, does not save model checkpoint.')
parser.add_argument('-rog', '--reset-grid-optim', type=bool, default=False,
                    help='If true, loads completely new params for the grid and optimizer.')
parser.add_argument('-is', '--image-size', type=int, default=40,
                    help='Size of each side of the image. Determines grid size.')
parser.add_argument('-sc', '--image-scaler', type=int, default=1,
                    help='How much to scale the entire simulation by (changes the dimensions of the model).')
args = parser.parse_args()

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

# Setup tensorboard
tb_parent_dir = './runs/'
repo = git.Repo(search_parent_directories=True)
sha = repo.head.object.hexsha
#head = repo.head
local_branch = repo.active_branch.name
run_dir = local_branch + '/' + sha[-3:] + '/' +  datetime.datetime.now().isoformat(timespec='seconds') + '/'
print('TB Log Directory is: ', tb_parent_dir + run_dir)
writer = SummaryWriter(log_dir=tb_parent_dir + run_dir)

# Setup model saving
model_parent_dir = './model_checkpoints/'
model_checkpoint_dir = model_parent_dir + local_branch + '/'
path = Path(model_checkpoint_dir)
path.mkdir(parents=True, exist_ok=True)

# ## Set Backend
backend_name = "torch"
#backend_name = "torch.float32"
#backend_name = "torch.float16"
#backend_name = "torch.cuda.float32"
#backend_name = "torch.cuda.float64"
fdtd.set_backend(backend_name)
if(backend_name.startswith("torch.cuda")):
    device = "cuda"
else:
    device = "cpu"

image_transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.RandomVerticalFlip(p=0.5),
    torchvision.transforms.RandomHorizontalFlip(p=0.5),
    #torchvision.transforms.RandomRotation(degrees=[0, 360], expand=True),
    RandomRot90(),
    torchvision.transforms.ColorJitter(brightness=0.5, hue=0.3),
    torchvision.transforms.RandomInvert(p=0.5),
    torchvision.transforms.Resize((args.image_size, args.image_size))])
train_dataset = torchvision.datasets.Flowers102('flowers102/', 
                                           split='train',
                                           download=True,
                                           transform=image_transform)
#TODO - turn SHUFFLE back to TRUE for training on multiple images.
train_loader = torch.utils.data.DataLoader(train_dataset,
                                           batch_size=1, 
                                           shuffle=True)


sample = get_sample_img(train_loader, color=True)
print('Image shape: ', sample.shape)
ih, iw = tuple(sample.shape[2:4])

# Physics constants
WAVELENGTH = 1550e-9 # meters
SPEED_LIGHT: float = 299_792_458.0  # [m/s] speed of light
GRID_SPACING = 0.1 * WAVELENGTH # meters



# Size of grid boundary layer
bw = 10*args.image_scaler
# Create FDTD Grid
grid_h, grid_w = (ih*args.image_scaler+bw*2, iw*args.image_scaler+bw*2)
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
#em_steps = 283


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
if((grid_path is not None) and (not args.reset_grid_optim)):
    print('Loading grid params...')
    with torch.no_grad():
        load_grid_params_to_learn = torch.load(grid_path)
        for idx, tensor in enumerate(load_grid_params_to_learn):
            if(args.image_scaler == 1 or tensor.shape == grid_params_to_learn[idx][...].shape):
                grid_params_to_learn[idx][...] = tensor[...]
            else:
                if(idx == len(load_grid_params_to_learn) - 1):
                    tensor = torch.squeeze(tensor)
                # Interpolate the thing....
                print('INFO: grid is being scaled. Shapes are mismatched: {0} vs {1}'.format(tensor[...].shape, grid_params_to_learn[idx][...].shape))
                
                # If this is a grid param, expand it over the spatial dims.
                if(len(tensor.shape) > 1):
                    reps = torch.ones(len(tensor.shape), dtype=int)
                    for i in range(len(reps)):
                        reps[i] = args.image_scaler
                        if(i >= 1):
                            break

                    print('Reps: ', reps)
                    grid_params_to_learn[idx][...] = tensor.repeat(tuple(reps))
                    print('New shape: ', grid_params_to_learn[idx][...].shape)
                # If this is the loss step weights, scale it linearly to fit the new size.
                else:
                    x = torch.nn.functional.interpolate(tensor[None, None, ...], grid_params_to_learn[idx][...].shape, mode='linear')
                    print(x.shape)

                # Since parameter shapes have changed, the optimizer weights are obsolete.
                reset_optimizer = True

# Combine grid and model params and register them with the optimizer.
params_to_learn = [*model.parameters()] + grid_params_to_learn
optimizer = optim.AdamW(params_to_learn, lr=0.0001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.01, amsgrad=False)
if((not reset_optimizer) and (optimizer_path is not None) and (not args.reset_grid_optim)):
    print('Loading optimizer params...')
    optimizer.load_state_dict(torch.load(optimizer_path))
else:
    print('Starting with a fresh optimizer.')

mse = torch.nn.MSELoss(reduce=False)
loss_fn = torch.nn.MSELoss()

grid.H.requires_grad = True
grid.H.retain_grad()
grid.E.requires_grad = True
grid.E.retain_grad()

# For timing steps
stopwatch = time.time()

# Train the weights
for train_step in range(start_step + 1, start_step + args.max_steps):
    # Generate a new image
    img = get_sample_img(train_loader, color=True)

    # Reset grid and optimizer
    grid.reset()
    optimizer.zero_grad()

    loss_list = []
    # Get sample from training data
    em_step_loss_weight_dist = softmax(torch.squeeze(loss_step_weights))
    argmax_step = torch.argmax(torch.squeeze(loss_step_weights))
    for em_step, (img_hat_em, em_field) in enumerate(model(img)):
        loss_list += [loss_fn(img_hat_em[None, ...], img)]
        if(em_step == argmax_step):
            e_field_img = em_field[0:3,...]
            h_field_img = em_field[3:6,...]
            img_hat_em_save = img_hat_em
    loss_per_step = torch.stack(loss_list)
    weighted_loss_per_step = loss_per_step * em_step_loss_weight_dist
    loss = torch.sum(weighted_loss_per_step)

    # Add the argmaxxed images to tensorboard
    img_grid = torchvision.utils.make_grid([img[0,...], img_hat_em_save,
        norm_img_by_chan(e_field_img), 
        norm_img_by_chan(h_field_img)])
    writer.add_image('sample', img_grid, train_step)

    perm = torch.reshape(get_object_by_name(grid, 'cc_substrate').inverse_permittivity, (-1, iw, ih))
    writer.add_image('ccsubstrate1', perm[0:3,...], train_step)
    writer.add_image('ccsubstrate2', perm[3:6,...], train_step)
    writer.add_image('ccsubstrate3', perm[6:9,...], train_step)


    writer.add_scalar('Total Loss', loss, train_step)
    writer.add_histogram('Loss Per Step', loss_per_step, train_step)
    writer.add_histogram('Weighted Loss Per Step', weighted_loss_per_step, train_step)
    writer.add_histogram('Loss EM Step Weights', loss_step_weights, train_step)
    writer.add_scalar('em_steps', em_steps, train_step)
    writer.add_scalar('ccsubstate_sum', 
            torch.sum(get_object_by_name(grid, 'cc_substrate').inverse_permittivity), train_step)
    writer.add_scalar('Argmax EM Step', argmax_step, train_step)

    print('Step: ', train_step, '\tTime: ', grid.time_steps_passed, '\tLoss: ', loss)

    # Tensorboard
    writer.add_histogram('cc_dirs', model.cc_dirs, train_step)
    writer.add_histogram('cc_freqs', model.cc_freqs, train_step)
    writer.add_histogram('cc_phases', model.cc_phases, train_step)
    writer.add_histogram('cc_nonlin_w', get_source_by_name(grid, 'cc').nonlin_conv.weight, train_step)
    writer.add_histogram('cc_nonlin_b', get_source_by_name(grid, 'cc').nonlin_conv.bias, train_step)
    writer.add_histogram('ccsubstrate', get_object_by_name(grid, 'cc_substrate').inverse_permittivity, train_step)
    writer.add_histogram('ccsubstrate_nonlin_w', get_object_by_name(grid, 'cc_substrate').nonlin_conv.weight, train_step)
    writer.add_histogram('ccsubstrate_nonlin_b', get_object_by_name(grid, 'cc_substrate').nonlin_conv.bias, train_step)
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
    if(not args.dry_run):
        if((train_step % args.save_steps == 0) and (train_step > 0)):
            torch.save(model.state_dict(), model_checkpoint_dir + 'md_'+str(train_step).zfill(12)+'.pt')
            torch.save(optimizer.state_dict(), model_checkpoint_dir + 'md_'+str(train_step).zfill(12)+'.opt')
            torch.save(grid_params_to_learn, model_checkpoint_dir + 'md_'+str(train_step).zfill(12)+'.grd')

    # Profile performance
    seconds_per_step = time.time() - stopwatch 
    writer.add_scalar('seconds_per_step', torch.tensor(seconds_per_step), train_step)
    stopwatch = time.time()

writer.close()
