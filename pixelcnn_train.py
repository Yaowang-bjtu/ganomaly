import numpy as np
import torch
import torch.nn.functional as F

from options import Options

from pixelcnn import GatedPixelCNN

global_steps = 0

def train(data_loader, model, prior, optimizer, device, writer):
    for images in data_loader:
        with torch.no_grad():
            images = images.to(device)
            latents = model(images)['vq_output']['encoding_indices']
            latents = latents.detach()

        
        logits = prior(latents, torch.tensor(0).to(device))
        logits = logits.permute(0, 2, 3, 1).contiguous()

        optimizer.zero_grad()
        loss = F.cross_entropy(logits.view(-1, 512),
                               latents.view(-1))
        loss.backward()

        # Logs
        writer.add_scalar('loss/train', loss.item(), args.steps)

        optimizer.step()
        global_steps += 1

def test(data_loader, model, prior, args, writer):
    with torch.no_grad():
        loss = 0.
        for images in data_loader:
            images = images.to(args.device)

            latents = model(images)['vq_output']['encoding_indices']
            latents = latents.detach()
            logits = prior(latents, torch.tensor(0).to(device))
            logits = logits.permute(0, 2, 3, 1).contiguous()
            loss += F.cross_entropy(logits.view(-1, args.k),
                                    latents.view(-1))

        loss /= len(data_loader)

    # Logs
    writer.add_scalar('loss/valid', loss.item(), args.steps)

    return loss.item()

def main(args):
    writer = SummaryWriter('./logs/{0}'.format('pixelcnn'))
    save_filename = './models/{0}/prior.pt'.format('pixelcnn')

    batch_size = 32
    image_size = 32

    opt = Options().parse()
    dataset = "NanjingRail_blocks2"
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
    device = torch.device("cuda1" if torch.cuda.is_available() else "cpu")
    model = vqvaemodel.cuda(1)
    path_vq = './models/vqvae_anomaly.pth'
    pretrained_data = torch.load(data_path)
    model.load_state_dict(pretrained_data)
    model.eval()


    prior = GatedPixelCNN(512, 64,
        15, n_classes=1).to(device)
    optimizer = torch.optim.Adam(prior.parameters(), lr=3e-4)

    best_loss = -1.
    for epoch in range(10):
        train(train_dataset, model, prior, optimizer, device, writer)
        # The validation loss is not properly computed since
        # the classes in the train and valid splits of Mini-Imagenet
        # do not overlap.
        loss = test(valid_loader, model, prior, device, writer)

        if (epoch == 0) or (loss < best_loss):
            best_loss = loss
            with open(save_filename, 'wb') as f:
                torch.save(prior.state_dict(), f)

if __name__ == '__main__':


    main()
