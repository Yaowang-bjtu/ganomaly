# alpha-GAN model

import numpy as np
import pandas as pd
import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import init, Parameter
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torchvision import datasets, models, transforms
from torchvision.datasets import ImageFolder

from alphagan import AlphaGAN

from options import Options
from lib.data import load_data
from psutil import cpu_count
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
import pdb

class ChannelsToLinear(nn.Linear):
    """Flatten a Variable to 2d and apply Linear layer"""
    def forward(self, x):
        b = x.size(0)
        return super().forward(x.view(b,-1))

class LinearToChannels2d(nn.Linear):
    """Reshape 2d Variable to 4d after Linear layer"""
    def __init__(self, m, n, w=1, h=None, **kw):
        h = h or w
        super().__init__(m, n*w*h, **kw)
        self.w = w
        self.h = h
    def forward(self, x):
        b = x.size(0)
        return super().forward(x).view(b, -1, self.w, self.h)

class ResBlock(nn.Module):
    """Simple ResNet block"""
    def __init__(self, c,
                 activation=nn.LeakyReLU, norm=nn.BatchNorm2d,
                 init_gain=1, groups=1):
        super().__init__()
        self.a1 = activation()
        self.a2 = activation()
        self.norm1 = norm and norm(c)
        self.norm2 = norm and norm(c)
        
        to_init = []
        self.conv1 = nn.Conv2d(
            c, c, 3, 1, 1, bias=bool(norm), groups=groups)
        to_init.append(self.conv1.weight)            
        self.conv2 = nn.Conv2d(
            c, c, 3, 1, 1, bias=bool(norm), groups=groups)
        to_init.append(self.conv2.weight)
        
        # if using grouping, add a 1x1 convolution to each conv layer
        if groups!=1:
            self.conv1 = nn.Sequential(
                self.conv1, nn.Conv2d(c,c,1,bias=bool(norm)))
            self.conv2 = nn.Sequential(
                self.conv2, nn.Conv2d(c,c,1,bias=bool(norm)))
            to_init.extend([self.conv1[1].weight, self.conv2[1].weight])
                    
        # init
        for w in to_init:
            init.xavier_normal(w, init_gain)
        
    def forward(self, x):
        y = self.conv1(x)
        if self.norm1:
            y = self.norm1(y)
        y = self.a1(y)
        
        y = self.conv2(y)
        if self.norm2:
            y = self.norm2(y)
                
        return self.a2(x+y)

latent_dim = 128
batch_size = 64
use_gpu = True

# encoder network
h = 128
resample = nn.AvgPool2d
norm = nn.BatchNorm2d#None
a, g = nn.ReLU, init.calculate_gain('relu')
groups = 1#h//8
E = nn.Sequential(
    nn.Conv2d(3,h,5,1,2), resample(2), a(),
    ResBlock(h, activation=a, norm=norm, init_gain=g, groups=groups), resample(2),
    ResBlock(h, activation=a, norm=norm, init_gain=g, groups=groups), resample(2),
    ResBlock(h, activation=a, norm=norm, init_gain=g, groups=groups),
    ChannelsToLinear(h*16, latent_dim)
)
for layer in (0,8):
    init.xavier_normal(E[layer].weight, g)

t = Variable(torch.randn(batch_size,3,32,32))
assert E(t).size() == (batch_size,latent_dim)


# generator network
h = 128
norm = nn.BatchNorm2d#None
a, g = nn.ReLU, init.calculate_gain('relu')
groups = 1#h//8
resample = lambda x: nn.Upsample(scale_factor=x)
G = nn.Sequential(
    LinearToChannels2d(latent_dim,h,4,4), a(),
    ResBlock(h, activation=a, norm=norm, init_gain=g, groups=groups), resample(2),
    ResBlock(h, activation=a, norm=norm, init_gain=g, groups=groups), resample(2),
    ResBlock(h, activation=a, norm=norm, init_gain=g, groups=groups), resample(2),
    ResBlock(h, activation=a, norm=norm, init_gain=g, groups=groups),
    nn.Conv2d(h, 3, 1), nn.Tanh()
)
for layer in (0,9):
    init.xavier_normal(G[layer].weight, g)

t = Variable(torch.randn(batch_size,latent_dim))
assert G(t).size() == (batch_size,3,32,32)


# discriminator network
h = 128
resample = nn.AvgPool2d
norm = nn.BatchNorm2d
a, g = lambda: nn.LeakyReLU(.2), init.calculate_gain('leaky_relu', .2)
groups = 1

D = nn.Sequential(
    nn.Conv2d(3,h,5,1,2), resample(2), a(),
    ResBlock(h, activation=a, norm=norm, init_gain=g, groups=groups), resample(2),
    ResBlock(h, activation=a, norm=norm, init_gain=g, groups=groups), resample(2),
    ResBlock(h, activation=a, norm=norm, init_gain=g, groups=groups),
    ChannelsToLinear(h*16, 1), nn.Sigmoid()
)
for layer in (0,8):
    init.xavier_normal(D[layer].weight, g)
    
t = Variable(torch.randn(batch_size,3,32,32))
assert D(t).size() == (batch_size,1)


# code discriminator network
# no batch norm in the code discriminator, it causes trouble
h = 700
a, g = lambda: nn.LeakyReLU(.2), init.calculate_gain('leaky_relu', .2)
C = nn.Sequential(
    nn.Linear(latent_dim, h), a(),
    nn.Linear(h, h), a(),
    nn.Linear(h, 1), nn.Sigmoid(),
)

for i,layer in enumerate(C):
    if i%2==0:
        init.xavier_normal(layer.weight, g)

t = Variable(torch.randn(batch_size,latent_dim))
assert C(t).size() == (batch_size,1)


model = AlphaGAN(E, G, D, C, latent_dim, lambd=40, z_lambd=0)
if use_gpu:
    model = model.cuda()

diag = []
def log_fn(d):
    d = pd.DataFrame(d)
    diag.append(d)
    print(d)

def checkpoint_fn(model, epoch):
    path =  './models/cifar10_dim_{}_lambda_{}_zlambd_{}_epochs_{}.torch'.format(
        model.latent_dim, model.lambd, model.z_lambd, epoch
    )
    torch.save(model.state_dict(), path)

def torchvision_dataset(dset_fn, train=True):
    dset = dset_fn(
        data_dir,
        train=train,
        transform=transforms.Compose([
            transforms.ToTensor()
        ]),
        target_transform=None,
        download=True)
    return torch.stack(list(zip(*dset))[0])*2-1

if __name__ == "__main__":
    opt = Options().parse()
    dataset = "NanjingRail_blocks2"
    opt.batch_size = batch_size
    opt.isize = 32
    # dataloader = load_data(opt)

    transform = transforms.Compose([transforms.Scale(opt.isize),
                                        transforms.CenterCrop(opt.isize),
                                        transforms.ToTensor(),
                                        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)), ])

    dataset_train = ImageFolder('./data/{}/train'.format(dataset),transform)
    dataset_train = torch.stack(list(zip(*dataset_train))[0])
    dataset_test = ImageFolder('./data/{}/test'.format(dataset),transform)
    dataset_test = torch.stack(list(zip(*dataset_test))[0])

    X_train = DataLoader(dataset=dataset_train,
                         batch_size=opt.batchsize,
                         shuffle=True,
                         num_workers=int(opt.workers),
                         drop_last=False)
    
    X_test = DataLoader(dataset=dataset_train,
                         batch_size=opt.batchsize,
                         shuffle=False,
                         num_workers=int(opt.workers),
                         drop_last=False)


    # X_train = dataloader['train']
    # X_test = dataloader['test']


    # data_dir = './data'
    # cifar = torchvision_dataset(datasets.CIFAR10, train=True)
    # cifar_test = torchvision_dataset(datasets.CIFAR10, train=False)
    # num_workers = cpu_count() if use_gpu else 0

    # X_train = DataLoader(cifar, batch_size=batch_size, shuffle=True,
    #                  num_workers=num_workers, pin_memory=use_gpu)
    # X_test = DataLoader(cifar_test, batch_size=batch_size, shuffle=False,
    #                num_workers=num_workers, pin_memory=use_gpu)

    
    model.fit(
        X_train, X_test,
        n_iter=(2,1,1), n_epochs=100,
        log_fn=log_fn, log_every=1,
        checkpoint_fn=checkpoint_fn, checkpoint_every=2
    )

    model.eval()

    z, x = model(128, mode='sample')
    fig, ax = plt.subplots(1,1,figsize=(16,12))
    ax.imshow(make_grid(
        x.data, nrow=16, range=(-1,1), normalize=True
    ).cpu().numpy().transpose(1,2,0), interpolation='nearest')

    plt.show()
