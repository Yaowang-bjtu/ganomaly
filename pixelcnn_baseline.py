import torch
import torch.nn as nn
from torchvision import datasets, transforms
from pixelcnn import GatedPixelCNN
import numpy as np
from torchvision.utils import save_image
import time
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from sklearn import metrics

import matplotlib.pyplot as plt

#RUN = 'test' #| "train"


BATCH_SIZE = 32
N_EPOCHS = 10
PRINT_INTERVAL = 100
ALWAYS_SAVE = False
DATASET = 'RAIL'  # CIFAR10 | MNIST | FashionMNIST
#DATASET = 'FashionMNIST'  # CIFAR10 | MNIST | FashionMNIST

NUM_WORKERS = 4

IMAGE_SHAPE = (32, 32)  # (32, 32) | (28, 28)
INPUT_DIM = 3  # 3 (RGB) | 1 (Grayscale)
K = 256
DIM = 64
N_LAYERS = 15
LR = 3e-4


model = GatedPixelCNN(K, DIM, N_LAYERS,n_classes=1).cuda()
criterion = nn.CrossEntropyLoss().cuda()
opt = torch.optim.Adam(model.parameters(), lr=LR)


def train(train_loader):
    train_loss = []
    for batch_idx, (x, label) in enumerate(train_loader):
        start_time = time.time()
        x = (x[:, 0] * (K-1)).long().cuda()
        label = label.cuda()

        # Train PixelCNN with images
        logits = model(x, label)
        logits = logits.permute(0, 2, 3, 1).contiguous()

        loss = criterion(
            logits.view(-1, K),
            x.view(-1)
        )

        opt.zero_grad()
        loss.backward()
        opt.step()

        train_loss.append(loss.item())

        if (batch_idx + 1) % PRINT_INTERVAL == 0:
            print('\tIter: [{}/{} ({:.0f}%)]\tLoss: {} Time: {}'.format(
                batch_idx * len(x), len(train_loader.dataset),
                PRINT_INTERVAL * batch_idx / len(train_loader),
                np.asarray(train_loss)[-PRINT_INTERVAL:].mean(0),
                time.time() - start_time
            ))


def test(test_loader):
    start_time = time.time()
    val_loss = []
    with torch.no_grad():
        for batch_idx, (x, label) in enumerate(test_loader):
            x = (x[:, 0] * (K-1)).long().cuda()
            label = label.cuda()

            logits = model(x, label)
            logits = logits.permute(0, 2, 3, 1).contiguous()
            loss = criterion(
                logits.view(-1, K),
                x.view(-1)
            )
            val_loss.append(loss.item())

    print('Validation Completed!\tLoss: {} Time: {}'.format(
        np.asarray(val_loss).mean(0),
        time.time() - start_time
    ))
    return np.asarray(val_loss).mean(0)


def generate_samples():
    #label = torch.arange(10).expand(10, 10).contiguous().view(-1)
    label = torch.zeros((3,3)).view(-1)
    label = label.long().cuda()

    x_tilde = model.generate(label, shape=IMAGE_SHAPE, batch_size=9)
    images = x_tilde.cpu().data.float() / (K - 1)

    save_image(
        images[:, None],
        'samples/pixelcnn_baseline_samples_{}.png'.format(DATASET),
        nrow=3
    )




def main(ch, RUN = 'train'):

    # dataset_name = "NanjingRail_blocks1"
    dataset_name = "RailAnormaly_blocks{:02d}".format(ch)
    if DATASET == "RAIL":
        transform = transforms.Compose([transforms.Scale(IMAGE_SHAPE[0]),
                                                transforms.CenterCrop(IMAGE_SHAPE[0]),
                                                transforms.ToTensor(), ])

        dataset_train = ImageFolder('./data/{}/train'.format(dataset_name),transform)
        #dataset_train = torch.stack(list(zip(*dataset_train))[0])
        dataset_test = ImageFolder('./data/{}/test'.format(dataset_name),transform)
        #dataset_test = torch.stack(list(zip(*dataset_test))[0])

        train_loader = DataLoader(dataset=dataset_train,
                                batch_size=BATCH_SIZE,
                                shuffle=True,
                                num_workers=int(NUM_WORKERS),
                                drop_last=False)
        
        test_loader = DataLoader(dataset=dataset_test,
                                batch_size=BATCH_SIZE,
                                shuffle=False,
                                num_workers=int(NUM_WORKERS),
                                drop_last=False)

    else:
        train_loader = torch.utils.data.DataLoader(
            eval('datasets.'+DATASET)(
                '../data/{}/'.format(DATASET), train=True, download=True,
                transform=transforms.ToTensor(),
            ), batch_size=BATCH_SIZE, shuffle=False,
            num_workers=NUM_WORKERS, pin_memory=True
        )
        test_loader = torch.utils.data.DataLoader(
            eval('datasets.'+DATASET)(
                '../data/{}/'.format(DATASET), train=False,
                transform=transforms.ToTensor(),
            ), batch_size=BATCH_SIZE, shuffle=False,
            num_workers=NUM_WORKERS, pin_memory=True
        )

    if RUN == 'test':
        model.load_state_dict(torch.load('models/{}_pixelcnnBase.pt'.format(dataset_name)))
        #generate_samples()
        #(x,label) = list(test_loader)[-3]
        normal_likelihood = []
        abnormal_likelihood = []
        for x, label in test_loader:
            x = (x[:, 0] * (K-1)).long().cuda()
            #print(label)
            label_in = torch.zeros(len(label)).long().cuda() 
            #print(label_in)
            res = model.likelihood(x,label_in)
            normal_res = res[label == 0]
            normal_likelihood.append(normal_res.cpu().detach().numpy())
            abnormal_res = res[label == 1]
            abnormal_likelihood.append(abnormal_res.cpu().detach().numpy())
            #print(res)
        #print(np.mean(np.hstack(normal_likelihood)))
        #print(np.mean(np.hstack(abnormal_likelihood)))

        normal_likelihood = np.hstack(normal_likelihood)
        normal_label = np.zeros(normal_likelihood.size)
        abnormal_likelihood = np.hstack(abnormal_likelihood)
        abnormal_label = np.ones(abnormal_likelihood.size)

        y = np.hstack([normal_label,abnormal_label])
        scorce = np.hstack([normal_likelihood,abnormal_likelihood])

        fpr, tpr, _ = metrics.roc_curve(y, scorce)
        AUC = metrics.auc(fpr, tpr)
        print('C000{}: AUC = {}'.format(ch,AUC))

        # plt.hist([normal_likelihood,abnormal_likelihood],bins=70,density=True)
        # plt.figure()
        # plt.plot(fpr,tpr)
        # plt.show()

        # for res in normal_likelihood:
        #     res_np = res.cpu().detach().numpy()

    else:
        BEST_LOSS = 999
        LAST_SAVED = -1
        for epoch in range(1, N_EPOCHS):
            print("\nEpoch {}:".format(epoch))
            train(train_loader)
            cur_loss = test(test_loader)

            if ALWAYS_SAVE or cur_loss <= BEST_LOSS:
                BEST_LOSS = cur_loss
                LAST_SAVED = epoch

                print("Saving model!")
                torch.save(model.state_dict(), 'models/{}_pixelcnnBase.pt'.format(dataset_name))
            else:
                print("Not saving model! Last saved: {}".format(LAST_SAVED))

            #generate_samples()


if __name__ == '__main__':
    for i in range(1,9):
        main(i,'train')

    for i in range(1,9):
        main(i,'test')
