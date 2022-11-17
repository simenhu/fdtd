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

## Create dataloader, in PyTorch, we feed the trainer data with use of dataloader
## We create dataloader with dataset from torchvision, 
## and we dont have to download it seperately, all automatically done

# Define batch size, batch size is how much data you feed for training in one iteration
batch_size_train = 64 # We use a small batch size here for training
batch_size_test = 1024 #

# define how image transformed
image_transform = torchvision.transforms.Compose([
                               torchvision.transforms.ToTensor()])
train_dataset = torchvision.datasets.CIFAR10('cifar10/', 
                                           train=True, 
                                           download=True,
                                           transform=image_transform)
test_dataset = torchvision.datasets.CIFAR10('cifar10/', 
                                           train=False, 
                                           download=True,
                                           transform=image_transform)
#data loaders
train_loader = torch.utils.data.DataLoader(train_dataset,
                                           batch_size=batch_size_train, 
                                           shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset,
                                          batch_size=batch_size_test, 
                                          shuffle=True)


# We can check the dataloader
_, (example_datas, labels) = next(enumerate(test_loader))
sample = example_datas[0][0]
# show the data
plt.imshow(sample, cmap='gray', interpolation='none')
print("Label: "+ str(labels[0]))


## Then define the model class
class WaveAutoEncoder(nn.Module):
    def __init__(self, input_chans=3, em_chans=3, load_weights=False):
        super(AutoEncoder, self).__init__()

        if(load_weights):
            #TODO - implement loading from pre-trained weights
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

        # EM Decoder layers.
        #TODO - find out how many layers the EM field makes up and replace it below.
        self.dec_conv1 = nn.Conv2d(emnumlayers, 16, kernel_size=5, stride=1, padding='same')
        self.dec_conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=1, padding='same')
        self.dec_conv3 = nn.Conv2d(32, 16, kernel_size=5, stride=1, padding='same')
        self.dec_conv4 = nn.Conv2d(16, ic, kernel_size=5, stride=1, padding='same')

        # The rise intensity of the signal (how fast it fires).
        self.frequencies = (oc)
        # The magnitude of the charge that is moved (how much charge to move between cells).
        self.magnitudes = (oc)
        # The direction of the charge when fired (vector in 3D).
        self.direction = (oc, 3)

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
        x = torch.sigmoid(x)

        # Convert activations to EM perturbation magnitudes with current E or H fields as inputs.
        mags = f(E_field, x)

        # Setup the simulation w/ the source mags

        # Run the EM simulation for N steps and get the resulting E and H fields
        E, H = sim.run(self.num_EM_steps)

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

## create model and optimizer
learning_rate = 0.00001
momentum = 0.5
device = "cuda"
model = AutoEncoder().to(device) #using cpu here
optimizer = optim.SGD(model.parameters(), lr=learning_rate,
                      momentum=momentum)
mse = torch.nn.MSELoss(reduce=False)


def train(model, device, train_loader, optimizer, epoch, log_interval=10000):
    model.train()
    tk0 = tqdm(train_loader, total=int(len(train_loader)))
    counter = 0
    for batch_idx, (data, target) in enumerate(tk0):
        data, label = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = torch.sum(mse(output, data))
        loss.backward()
        optimizer.step()
        counter += 1
        tk0.set_postfix(loss=(loss.item()*data.size(0) / (counter * train_loader.batch_size)))

def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for idx, (data, target) in enumerate(test_loader):
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += torch.sum(mse(output, data))
            # Display the images
            img = format_imgs(data, output)
            if(idx == 0):
                plt.imshow(img)
                plt.show()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


num_epoch = 10
for epoch in range(1, num_epoch + 1):
        test(model, device, test_loader)
        train(model, device, train_loader, optimizer, epoch)

