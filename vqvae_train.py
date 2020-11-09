from vqvae import Decoder, Encoder, VectorQuantizer, VQVAEModel
import torch
import torch.nn as nn
from torch import optim
from torch.utils.tensorboard import SummaryWriter
from options import Options
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
import numpy as np
import matplotlib.pyplot as plt

#!!!!!!!! select correct dataset here!!!!!!!!
DATASET = 'C0003' #dataset

def main():

  writer = SummaryWriter('runs1/vqvae_train')
  #path = './models/vqvae_anomaly02.pth'
  path = './models/vqvae_anomaly{}.pth'.format(DATASET[3:])
  

  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

  # Set hyper-parameters.
  batch_size = 32
  image_size = 32

  opt = Options().parse()
  #dataset = "RailAnormaly_blocks02"
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


  train_data_variance = np.var(dataset_train.data.numpy())
  print(train_data_variance)

  # train_dataset = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
  #                                         shuffle=True)

  # testset = torchvision.datasets.CIFAR10(root='./data', train=False,
  #                                      download=True, transform=transform)
  # test_dataset = torch.utils.data.DataLoader(testset, batch_size=batch_size,
  #                                        shuffle=False)

  # classes = ('plane', 'car', 'bird', 'cat',
  #          'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

  # def imshow(img):
  #   img = img / 1 + 0.5     # unnormalize
  #   npimg = img.numpy()
  #   plt.imshow(np.transpose(npimg, (1, 2, 0)))
  #   plt.show()

  # get some random training images
  # dataiter = iter(trainloader)
  # images, labels = dataiter.next()

  # show images
  # imshow(torchvision.utils.make_grid(images))
  # # print labels
  # print(' '.join('%5s' % classes[labels[j]] for j in range(4)))


  # 100k steps should take < 30 minutes on a modern (>= 2017) GPU.
  num_training_updates = 100000

  num_hiddens = 128
  num_residual_hiddens = 32
  num_residual_layers = 2

  # This value is not that important, usually 64 works.
  # This will not change the capacity in the information-bottleneck.
  embedding_dim = 64

  # The higher this value, the higher the capacity in the information bottleneck.
  num_embeddings = 512

  # commitment_cost should be set appropriately. It's often useful to try a couple
  # of values. It mostly depends on the scale of the reconstruction cost
  # (log p(x|z)). So if the reconstruction cost is 100x higher, the
  # commitment_cost should also be multiplied with the same amount.
  commitment_cost = 0.25

  learning_rate = 3e-4

  # # Build modules.
  encoder = Encoder(num_hiddens, num_residual_layers, num_residual_hiddens)
  decoder = Decoder(embedding_dim, num_hiddens, num_residual_layers, num_residual_hiddens)
  pre_vq_conv1 = nn.Conv2d(num_hiddens, embedding_dim, 1, 1)

  vq_vae = VectorQuantizer(
      embedding_dim=embedding_dim,
      num_embeddings=num_embeddings,
      commitment_cost=commitment_cost)
  
  model = VQVAEModel(encoder, decoder, vq_vae, pre_vq_conv1,
                   data_variance=train_data_variance)


  # train_batch = next(iter(train_dataset))
  # writer.add_graph(model,train_batch[0])

  model = model.cuda(1)
  optimizer = optim.Adam(model.parameters(),lr = learning_rate)



  train_losses = []
  train_recon_errors = []
  train_perplexities = []
  train_vqvae_loss = []

  def train_step(data):
    optimizer.zero_grad()
    model_output = model(data, is_training=True)
    loss = model_output['loss']
    loss.backward()
    optimizer.step()
    return model_output

  step_index = 0
  ex = 0
  for epoch in range(200):
    for data in train_dataset:
      train_results = train_step(data.cuda(1))
      step_index = step_index + 1
      train_losses.append(train_results['loss'].cpu().detach().numpy())
      train_recon_errors.append(train_results['recon_error'].cpu().detach().numpy())
      train_perplexities.append(train_results['vq_output']['perplexity'].cpu().detach().numpy())
      train_vqvae_loss.append(train_results['vq_output']['loss'].cpu().detach().numpy())

      if (step_index + 1) % 100 == 0:
        print('%d. train loss: %f ' % (step_index + 1,
                                      np.mean(train_losses[-100:])) +
              ('recon_error: %.3f ' % np.mean(train_recon_errors[-100:])) +
              ('perplexity: %.3f ' % np.mean(train_perplexities[-100:])) +
              ('vqvae loss: %.3f' % np.mean(train_vqvae_loss[-100:])))
        writer.add_histogram('Embeddings',vq_vae.embedding_layer.weight,step_index)
        writer.add_scalar('Loss/loss',train_results['loss'],step_index)
        writer.add_scalar('Loss/recon_error',train_results['recon_error'],step_index)

      if step_index >= num_training_updates:
        ex = 1
        break
    
    if ex:
      break
  # save model
 
  torch.save(model.state_dict(),path)

  # test the result
  f = plt.figure(figsize=(16,8))
  ax = f.add_subplot(1,2,1)
  ax.plot(train_recon_errors)
  ax.set_yscale('log')
  ax.set_title('NMSE.')

  ax = f.add_subplot(1,2,2)
  ax.plot(train_perplexities)
  ax.set_title('Average codebook usage (perplexity).')

  # Reconstructions
  train_batch = next(iter(train_dataset))
  valid_batch = next(iter(test_dataset))

  # Put data through the model with is_training=False, so that in the case of 
  # using EMA the codebook is not updated.
  train_reconstructions = model(train_batch.cuda(1),
                                is_training=False)['x_recon'].cpu().detach().numpy()
  valid_reconstructions = model(valid_batch.cuda(1),
                                is_training=False)['x_recon'].cpu().detach().numpy()


  def convert_batch_to_image_grid(image_batch):
    reshaped = (image_batch.transpose(0, 2, 3, 1)
                .reshape(4, 8, 32, 32, 3)
                .transpose(0, 2, 1, 3, 4)
                .reshape(4 * 32, 8 * 32, 3))
    return reshaped + 0.5

  f = plt.figure(figsize=(16,8))
  ax = f.add_subplot(2,2,1)
  ax.imshow(convert_batch_to_image_grid(train_batch.numpy()),
            interpolation='nearest')
  ax.set_title('training data originals')
  plt.axis('off')

  ax = f.add_subplot(2,2,2)
  ax.imshow(convert_batch_to_image_grid(train_reconstructions),
            interpolation='nearest')
  ax.set_title('training data reconstructions')
  plt.axis('off')

  ax = f.add_subplot(2,2,3)
  ax.imshow(convert_batch_to_image_grid(valid_batch.numpy()),
            interpolation='nearest')
  ax.set_title('validation data originals')
  plt.axis('off')

  ax = f.add_subplot(2,2,4)
  ax.imshow(convert_batch_to_image_grid(valid_reconstructions),
            interpolation='nearest')
  ax.set_title('validation data reconstructions')
  plt.axis('off')

  plt.show()

if __name__ == '__main__':
  main()