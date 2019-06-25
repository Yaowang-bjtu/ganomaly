from anomaly import model

import torch
import matplotlib.pyplot as plt
from options import Options
from torchvision import datasets, models, transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from torchvision.utils import make_grid

path = './models/cifar10_dim_128_lambda_40_zlambd_0_epochs_50.torch'

state_dict = torch.load(path)

model.load_state_dict(state_dict)
model.eval()

batch_size = 64
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

z, x = model(128, mode='sample')
fig, ax = plt.subplots(1,1,figsize=(16,12))
ax.imshow(make_grid(
    x.data, nrow=16, range=(-1,1), normalize=True
).cpu().numpy().transpose(1,2,0), interpolation='nearest')

plt.show()

