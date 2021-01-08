import numpy as np
import torch
import torch.nn.functional as F

from options import Options
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from pixelcnn import GatedPixelCNN
from torch.utils.tensorboard import SummaryWriter
from sklearn import metrics

import matplotlib.pyplot as plt

global_steps = 0
#!!!!!!!! select correct dataset here!!!!!!!!
DATASET = 'C0007'

def train(data_loader, model, prior, optimizer, device, writer):
    global global_steps
    for images in data_loader:
        with torch.no_grad():
            images = images.to(device)
            latents = model(images)['vq_output']['encoding_indices']
            latents = latents.detach()

        
        logits = prior(latents, torch.zeros(latents.shape[0]).long().to(device))
        logits = logits.permute(0, 2, 3, 1).contiguous()

        optimizer.zero_grad()
        loss = F.cross_entropy(logits.view(-1, 512),
                               latents.view(-1))
        loss.backward()

        # Logs
        writer.add_scalar('loss/train', loss.item(), global_steps)

        optimizer.step()
        global_steps = global_steps + 1

def test(data_loader, model, prior, device, writer):
    with torch.no_grad():
        loss = 0.
        for images in data_loader:
            images = images.to(device)

            latents = model(images)['vq_output']['encoding_indices']
            latents = latents.detach()
            logits = prior(latents, torch.zeros(latents.shape[0]).long().to(device))
            logits = logits.permute(0, 2, 3, 1).contiguous()
            loss += F.cross_entropy(logits.view(-1, 512),
                                    latents.view(-1))

        loss /= len(data_loader)

    # Logs
    writer.add_scalar('loss/valid', loss.item(), global_steps)

    return loss.item()

def main(args):
    writer = SummaryWriter('./logs/{0}'.format('pixelcnn'))
    save_filename = './models/{0}/prior{1}.pt'.format('pixelcnn',DATASET)

    batch_size = 32
    image_size = 32

    opt = Options().parse()
    dataset = "RailAnormaly_blocks{}".format(DATASET[3:])
    opt.batchsize = batch_size
    opt.isize = image_size

    # dataset
    transform = transforms.Compose([transforms.Scale(opt.isize),
                                            transforms.CenterCrop(opt.isize),
                                            transforms.ToTensor(),
                                            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)), ])

    dataset_train = ImageFolder('./data/{}/train'.format(dataset),transform)
    dataset_train = torch.stack(list(zip(*dataset_train))[0])
    dataset_test = ImageFolder('./data/{}/test'.format(dataset),transform)
    dataset_test = torch.stack(list(zip(*dataset_test))[0])

    train_dataset = DataLoader(dataset=dataset_train,
                            batch_size=opt.batchsize,
                            shuffle=True,
                            num_workers=int(opt.workers),
                            drop_last=False)
    
    test_dataset = DataLoader(dataset=dataset_train,
                            batch_size=opt.batchsize,
                            shuffle=False,
                            num_workers=int(opt.workers),
                            drop_last=False)

    from vqvae import vqvaemodel
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    model = vqvaemodel.cuda(1)
    path_vq = './models/vqvae_anomaly{}.pth'.format(DATASET[3:])
    pretrained_data = torch.load(path_vq)
    model.load_state_dict(pretrained_data)
    model.eval()


    prior = GatedPixelCNN(512, 64,
        15, n_classes=1).to(device)
    optimizer = torch.optim.Adam(prior.parameters(), lr=3e-4)

    best_loss = -1.
    for epoch in range(100):
        
        train(train_dataset, model, prior, optimizer, device, writer)
        # The validation loss is not properly computed since
        # the classes in the train and valid splits of Mini-Imagenet
        # do not overlap.
        
        loss = test(train_dataset, model, prior, device, writer)
        print("EPOCH{}, loss={}".format(epoch,loss))

        if (epoch == 0) or (loss < best_loss):
            best_loss = loss
            with open(save_filename, 'wb') as f:
                torch.save(prior.state_dict(), f)

def likelihood_test(DATASET):
    save_filename = './models/{0}/prior{1}.pt'.format('pixelcnn',DATASET)

    batch_size = 32
    image_size = 32

    opt = Options().parse()
    dataset = "RailAnormaly_blocks{}".format(DATASET[3:])
    opt.batchsize = batch_size
    opt.isize = image_size
    
    transform = transforms.Compose([transforms.Scale(opt.isize),
                                            transforms.CenterCrop(opt.isize),
                                            transforms.ToTensor(),
                                            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)), ])


    dataset_test = ImageFolder('./data/{}/test'.format(dataset),transform)
    test_dataset = DataLoader(dataset=dataset_test,
                            batch_size=opt.batchsize,
                            shuffle=False,
                            num_workers=int(opt.workers),
                            drop_last=False)    

    from vqvae import vqvaemodel
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    model = vqvaemodel.cuda(1)
    path_vq = './models/vqvae_anomaly{}.pth'.format(DATASET[3:])
    pretrained_data = torch.load(path_vq)
    model.load_state_dict(pretrained_data)
    model.eval()

    prior = GatedPixelCNN(512, 64,
        15, n_classes=1).to(device)    

    prior.load_state_dict(torch.load(save_filename))
    #generate_samples()
    #(x,label) = list(test_loader)[-3]
    normal_likelihood = []
    abnormal_likelihood = []
    for x, label in test_dataset:
        images = x.to(device)
        latents = model(images)['vq_output']['encoding_indices']
        latents = latents.detach()  
        res = prior.likelihood(latents,torch.zeros(latents.shape[0]).long().to(device))  
        #print(res)
        #print(label)
   
        normal_res = res[label == 0]
        normal_likelihood.append(normal_res.cpu().detach().numpy())
        abnormal_res = res[label == 1]
        abnormal_likelihood.append(abnormal_res.cpu().detach().numpy())
        #print(res)
    #print("normal likelihood: mean={0:.2f}, max={1:.2f}".format(np.mean(np.hstack(normal_likelihood)),
    #                                                        np.max(np.hstack(normal_likelihood))))
    #print("abnormal likelihood: mean={0:.2f}, min={1:.2f}".format(np.mean(np.hstack(abnormal_likelihood)),
    #                                                        np.min(np.hstack(abnormal_likelihood)))) 
    normal_likelihood = np.hstack(normal_likelihood)
    normal_label = np.zeros(normal_likelihood.size)
    abnormal_likelihood = np.hstack(abnormal_likelihood)
    abnormal_label = np.ones(abnormal_likelihood.size)

    y = np.hstack([normal_label,abnormal_label])
    scorce = np.hstack([normal_likelihood,abnormal_likelihood])

    fpr, tpr, _ = metrics.roc_curve(y, scorce)
    AUC = metrics.auc(fpr, tpr)
    print(AUC)

    return normal_likelihood, abnormal_likelihood, fpr, tpr 

    # plt.hist([normal_likelihood,abnormal_likelihood],bins=70,density=True)
    # plt.figure()
    # plt.plot(fpr,tpr)
    # plt.show()

def save_result(file, result):
    import pickle
    pickle.dump(result,open(file,'wb'))

def read_result(file):
    import pickle
    return pickle.load(open(file,'rb'))


def visulization(result):
    print(len(result))
    normal_likelihood = []
    abnormal_likelihood = []
    for nl, al, _, _ in result:
        normal_likelihood.append(nl)
        abnormal_likelihood.append(al)
    normal_likelihood = np.hstack(normal_likelihood)
    abnormal_likelihood = np.hstack(abnormal_likelihood)
    plt.hist([normal_likelihood,abnormal_likelihood],bins=70,density=True)

    #plt.show()
    plt.figure()
    for _, _, fpr, tpr in result:
        plt.plot(fpr,tpr)
    plt.show()

if __name__ == '__main__':

    # #main('')
    # result_list = []
    # for i in range(1,9):
    #     dataset = 'C{:04d}'.format(i)
    #     print(dataset)
    #     result = likelihood_test(dataset)
    #     result_list.append(result)

    # save_result('likelihood.pkl',result_list)
    result = read_result('likelihood.pkl')
    visulization(result)

    



    