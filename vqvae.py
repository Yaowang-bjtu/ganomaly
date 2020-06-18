import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
import numpy as np
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from torch.autograd import Function


import collections
from typing import Callable, Iterable, Mapping, Optional, Sequence, Text, Tuple, Union

from torch.utils.tensorboard import SummaryWriter


TensorLike = Union[np.ndarray, torch.Tensor]
FloatLike = Union[float, np.floating, TensorLike]

'''
class StopGrad(Function):
  @staticmethod
  def forward(ctx, inputs):
    size = torch.tensor(inputs.shape)
    ctx.save_for_backward(size)
    return inputs

  @staticmethod
  def backward(ctx, grad_output):
    size, = ctx.saved_tensors
    res = torch.zeros(list(size.numpy())).cuda(1)
    return res
'''

class StopGrad(Function):
  @staticmethod
  def forward(ctx, inputs):
    return inputs

  @staticmethod
  def backward(ctx, grad_output):
    return None

class VectorQuantizer(nn.Module):
  """pytorch module representing the VQ-VAE layer.
  """

  def __init__(self,
               embedding_dim: int,
               num_embeddings: int,
               commitment_cost: FloatLike,
               dtype: torch.dtype = torch.float32):
    """Initializes a VQ-VAE module.

    Args:
      embedding_dim: dimensionality of the tensors in the quantized space.
        Inputs to the modules must be in this format as well.
      num_embeddings: number of vectors in the quantized space.
      commitment_cost: scalar which controls the weighting of the loss terms
        (see equation 4 in the paper - this variable is Beta).
      dtype: dtype for the embeddings variable, defaults to tf.float32.
      name: name of the module.
    """
    super(VectorQuantizer, self).__init__()
    self.embedding_dim = embedding_dim
    self.num_embeddings = num_embeddings
    self.commitment_cost = commitment_cost

    embedding_shape = [num_embeddings, embedding_dim]
    # initializer = initializers.VarianceScaling(distribution='uniform')
    embeddings = torch.empty(embedding_shape,dtype=dtype).uniform_(-3,3)
    self.embedding_layer = torch.nn.Embedding(num_embeddings, embedding_dim, _weight=embeddings)

    # self.embeddings = torch.nn.Parameter(torch.empty(embedding_shape,dtype=dtype).uniform_(-3,3).cuda(1),requires_grad=True)
    # self.register_parameter('embeddings1',self.embeddings)

    # torch.nn.init.uniform_(tensor, a=0.0, b=1.0)

  def forward(self, inputs, is_training):
    """Connects the module to some inputs.

    Args:
      inputs: Tensor, final dimension must be equal to embedding_dim. All other
        leading dimensions will be flattened and treated as a large batch.
      is_training: boolean, whether this connection is to training data.

    Returns:
      dict containing the following keys and values:
        quantize: Tensor containing the quantized version of the input.
        loss: Tensor containing the loss to optimize.
        perplexity: Tensor containing the perplexity of the encodings.
        encodings: Tensor containing the discrete encodings, ie which element
        of the quantized space each input element was mapped to.
        encoding_indices: Tensor containing the discrete encoding indices, ie
        which element of the quantized space each input element was mapped to.
    """
    
    inputs = inputs.transpose(1,3)
    flat_inputs = torch.reshape(inputs, [-1, self.embedding_dim])

    embeddings = self.embedding_layer.weight.T

    distances = (
        torch.sum(flat_inputs**2, 1, keepdims=True) -
        2 * torch.matmul(flat_inputs, embeddings) +
        torch.sum(embeddings**2, 0, keepdims=True))

    encoding_indices = torch.argmax(-distances, 1)
    encodings = F.one_hot(encoding_indices,
                          self.num_embeddings)
    encodings = encodings.float()
    # NB: if your code crashes with a reshape error on the line below about a
    # Tensor containing the wrong number of values, then the most likely cause
    # is that the input passed in does not have a final dimension equal to
    # self.embedding_dim. Ideally we would catch this with an Assert but that
    # creates various other problems related to device placement / TPUs.
    encoding_indices = torch.reshape(encoding_indices, inputs.shape[:-1])
    # encoding_indices = torch.reshape(encoding_indices, torch.shape(inputs)[:-1])
    
    # quantized = self.quantize(encoding_indices)
    quantized = self.embedding_layer(encoding_indices)

    
    e_latent_loss = torch.mean((StopGrad.apply(quantized) - inputs)**2)
    q_latent_loss = torch.mean((quantized - StopGrad.apply(inputs))**2)
    loss = q_latent_loss + self.commitment_cost * e_latent_loss

    # Straight Through Estimator
    # quantized = inputs + tf.stop_gradient(quantized - inputs)
    quantized = inputs + StopGrad.apply(quantized - inputs)

    avg_probs = torch.mean(encodings, 0)
    perplexity = torch.exp(-torch.sum(avg_probs *
                                       torch.log(avg_probs + 1e-10)))

    quantized = quantized.transpose(1,3)
    return {
        'quantize': quantized,
        'loss': loss,
        'perplexity': perplexity,
        'encodings': encodings,
        'encoding_indices': encoding_indices,
        'distances': distances,
    }

class VectorQuantizerOld(nn.Module):
  """pytorch module representing the VQ-VAE layer.

  Implements the algorithm presented in
  'Neural Discrete Representation Learning' by van den Oord et al.
  https://arxiv.org/abs/1711.00937

  Input any tensor to be quantized. Last dimension will be used as space in
  which to quantize. All other dimensions will be flattened and will be seen
  as different examples to quantize.

  The output tensor will have the same shape as the input.

  For example a tensor with shape [16, 32, 32, 64] will be reshaped into
  [16384, 64] and all 16384 vectors (each of 64 dimensions)  will be quantized
  independently.

  Attributes:
    embedding_dim: integer representing the dimensionality of the tensors in the
      quantized space. Inputs to the modules must be in this format as well.
    num_embeddings: integer, the number of vectors in the quantized space.
    commitment_cost: scalar which controls the weighting of the loss terms (see
      equation 4 in the paper - this variable is Beta).
  """

  def __init__(self,
               embedding_dim: int,
               num_embeddings: int,
               commitment_cost: FloatLike,
               dtype: torch.dtype = torch.float32):
    """Initializes a VQ-VAE module.

    Args:
      embedding_dim: dimensionality of the tensors in the quantized space.
        Inputs to the modules must be in this format as well.
      num_embeddings: number of vectors in the quantized space.
      commitment_cost: scalar which controls the weighting of the loss terms
        (see equation 4 in the paper - this variable is Beta).
      dtype: dtype for the embeddings variable, defaults to tf.float32.
      name: name of the module.
    """
    super(VectorQuantizer, self).__init__()
    self.embedding_dim = embedding_dim
    self.num_embeddings = num_embeddings
    self.commitment_cost = commitment_cost

    embedding_shape = [embedding_dim, num_embeddings]
    # initializer = initializers.VarianceScaling(distribution='uniform')
    # self.embeddings = torch.empty(embedding_shape,dtype=dtype).uniform_(-3,3).cuda(1)

    self.embeddings = torch.nn.Parameter(torch.empty(embedding_shape,dtype=dtype).uniform_(-3,3).cuda(1),requires_grad=True)
    self.register_parameter('embeddings1',self.embeddings)

    # torch.nn.init.uniform_(tensor, a=0.0, b=1.0)

  def forward(self, inputs, is_training):
    """Connects the module to some inputs.

    Args:
      inputs: Tensor, final dimension must be equal to embedding_dim. All other
        leading dimensions will be flattened and treated as a large batch.
      is_training: boolean, whether this connection is to training data.

    Returns:
      dict containing the following keys and values:
        quantize: Tensor containing the quantized version of the input.
        loss: Tensor containing the loss to optimize.
        perplexity: Tensor containing the perplexity of the encodings.
        encodings: Tensor containing the discrete encodings, ie which element
        of the quantized space each input element was mapped to.
        encoding_indices: Tensor containing the discrete encoding indices, ie
        which element of the quantized space each input element was mapped to.
    """
    
    inputs = inputs.transpose(1,3)
    flat_inputs = torch.reshape(inputs, [-1, self.embedding_dim])

    distances = (
        torch.sum(flat_inputs**2, 1, keepdims=True) -
        2 * torch.matmul(flat_inputs, self.embeddings) +
        torch.sum(self.embeddings**2, 0, keepdims=True))

    encoding_indices = torch.argmax(-distances, 1)
    encodings = F.one_hot(encoding_indices,
                          self.num_embeddings)
    encodings = encodings.float()
    # NB: if your code crashes with a reshape error on the line below about a
    # Tensor containing the wrong number of values, then the most likely cause
    # is that the input passed in does not have a final dimension equal to
    # self.embedding_dim. Ideally we would catch this with an Assert but that
    # creates various other problems related to device placement / TPUs.
    encoding_indices = torch.reshape(encoding_indices, inputs.shape[:-1])
    # encoding_indices = torch.reshape(encoding_indices, torch.shape(inputs)[:-1])
    quantized = self.quantize(encoding_indices)

    
    e_latent_loss = torch.mean((StopGrad.apply(quantized) - inputs)**2)
    q_latent_loss = torch.mean((quantized - StopGrad.apply(inputs))**2)
    loss = q_latent_loss + self.commitment_cost * e_latent_loss

    # Straight Through Estimator
    # quantized = inputs + tf.stop_gradient(quantized - inputs)
    quantized = inputs + StopGrad.apply(quantized - inputs)

    avg_probs = torch.mean(encodings, 0)
    perplexity = torch.exp(-torch.sum(avg_probs *
                                       torch.log(avg_probs + 1e-10)))

    quantized = quantized.transpose(1,3)
    return {
        'quantize': quantized,
        'loss': loss,
        'perplexity': perplexity,
        'encodings': encodings,
        'encoding_indices': encoding_indices,
        'distances': distances,
    }

  def quantize(self, encoding_indices):
    """Returns embedding tensor for a batch of indices."""
    w = torch.transpose(self.embeddings, 1, 0)
    # TODO(mareynolds) in V1 we had a validate_indices kwarg, this is no longer
    # supported in V2. Are we missing anything here?
    return F.embedding(encoding_indices,w)

class ResidualBlock(nn.Module):
  def __init__(self, in_channel, num_hiddens,
    num_residual_hiddens):
    super(ResidualBlock,self).__init__()
    self._num_hiddens = num_hiddens
    self._num_residual_hiddens = num_residual_hiddens
    self._in_channel = in_channel

    self.conv3 = nn.Conv2d(in_channel,num_residual_hiddens,3,padding=1) 
    self.conv1 = nn.Conv2d(num_residual_hiddens,num_hiddens,1)
    
  def forward(self, inputs):
    h = inputs
    conv3_out = self.conv3(F.relu(h))
    conv1_out = self.conv1(F.relu(conv3_out))
    h = h + conv1_out
    return F.relu(h)

class ResidualStack(nn.Module):
  def __init__(self, num_hiddens,
    num_residual_layers, num_residual_hiddens):
    super(ResidualStack,self).__init__()
    self._num_hiddens = num_hiddens
    self._num_residual_layers = num_residual_layers
    self._num_residual_hiddens = num_residual_hiddens


    resbkls = collections.OrderedDict()

    for i in range(self._num_residual_layers):
      resbkls['resbkl{}'.format(i)] = ResidualBlock(self._num_hiddens,self._num_hiddens,self._num_residual_hiddens)

    self._residual_stack = nn.Sequential(resbkls)

  def forward(self, inputs):
    return self._residual_stack(inputs)

class Encoder(nn.Module):
  def __init__(self, num_hiddens, num_residual_layers, num_residual_hiddens):
    super(Encoder, self).__init__()
    self._num_hiddens = num_hiddens
    self._num_residual_layers = num_residual_layers
    self._num_residual_hiddens = num_residual_hiddens

    self._enc_1 = nn.Conv2d(3, self._num_hiddens // 2, 4, 2, padding=2)
    self._enc_2 = nn.Conv2d(self._num_hiddens // 2, self._num_hiddens, 4, 2, padding=1)
    self._enc_3 = nn.Conv2d(self._num_hiddens, self._num_hiddens, 3, 1, padding=1)
    
    # resbkls = collections.OrderedDict()
    # for i in range(self._num_residual_layers):
    #   resbkls['resbkl{}'.format(i)] = ResidualBlock(self._num_hiddens,self._num_hiddens,self._num_residual_hiddens)
    # self._residual_stack = nn.Sequential(resbkls)

    self._residual_stack = ResidualStack(self._num_hiddens,self._num_residual_layers,self._num_residual_hiddens)


  def forward(self, inputs):
    h = F.relu(self._enc_1(inputs))
    h = F.relu(self._enc_2(h))
    h = F.relu(self._enc_3(h))
    return self._residual_stack(h)

class Decoder(nn.Module):
  def __init__(self, num_inputs, num_hiddens, num_residual_layers, num_residual_hiddens):
    super(Decoder, self).__init__()
    self._num_hiddens = num_hiddens
    self._num_residual_layers = num_residual_layers
    self._num_residual_hiddens = num_residual_hiddens
    self._num_inputs = num_inputs

    self._dec_1 = nn.Conv2d(self._num_inputs, self._num_hiddens, 3, 1, padding=1)
    self._residual_stack = ResidualStack(self._num_hiddens,self._num_residual_layers, self._num_residual_hiddens)
    self._dec_2 = nn.ConvTranspose2d(self._num_hiddens, self._num_hiddens // 2, 4, 2, padding=1)
    self._dec_3 = nn.ConvTranspose2d(self._num_hiddens // 2, 3, 4, 2, padding=1)

  def forward(self, inputs):
    h = self._dec_1(inputs)
    h = self._residual_stack(h)
    h = F.relu(self._dec_2(h))
    x_recon = self._dec_3(h)
    return x_recon
    
class VQVAEModel(nn.Module):
  def __init__(self, encoder, decoder, vqvae, pre_vq_conv1, 
               data_variance):
    super(VQVAEModel, self).__init__()
    self._encoder = encoder
    self._decoder = decoder
    self._vqvae = vqvae
    self._pre_vq_conv1 = pre_vq_conv1
    self._data_variance = data_variance
  
  def forward(self, inputs, is_training=False):
    z = self._pre_vq_conv1(self._encoder(inputs))
    vq_output = self._vqvae(z, is_training=is_training)
    x_recon = self._decoder(vq_output['quantize'])
    recon_error = torch.mean((x_recon - inputs) ** 2) / self._data_variance
    loss = recon_error + vq_output['loss']
    return  {
        'z': z,
        'x_recon': x_recon,
        'loss': loss,
        'recon_error': recon_error,
        'vq_output': vq_output,
    }

# net = Encoder(10,2,12)
# #net = ResidualStack(10,2,12)
# print(net)
# params = list(net.parameters())
# print(len(params))
# print(params[0].size())

def createmodel():
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
                   data_variance=1)


  # train_batch = next(iter(train_dataset))
  # writer.add_graph(model,train_batch[0])

  model = model.cuda(1)

  return model

vqvaemodel = createmodel()


def main():

  writer = SummaryWriter('runs2/vqvae_train')
  path = './models/vqvae_New_None.pth'

  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

  # Set hyper-parameters.
  batch_size = 32
  image_size = 32

  # dataset
  transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (1, 1, 1))])

  trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)

  train_data_variance = np.var(trainset.data / 255.0)
  print(train_data_variance)

  train_dataset = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True)

  testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
  test_dataset = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                         shuffle=False)

  # classes = ('plane', 'car', 'bird', 'cat',
  #          'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

  def imshow(img):
    img = img / 1 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

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
  #optimizer = optim.SGD(model.parameters(),lr = learning_rate)



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
      train_results = train_step(data[0].cuda(1))
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
  train_reconstructions = model(train_batch[0].cuda(1),
                                is_training=False)['x_recon'].cpu().detach().numpy()
  valid_reconstructions = model(valid_batch[0].cuda(1),
                                is_training=False)['x_recon'].cpu().detach().numpy()


  def convert_batch_to_image_grid(image_batch):
    reshaped = (image_batch.transpose(0, 2, 3, 1)
                .reshape(4, 8, 32, 32, 3)
                .transpose(0, 2, 1, 3, 4)
                .reshape(4 * 32, 8 * 32, 3))
    return reshaped + 0.5

  f = plt.figure(figsize=(16,8))
  ax = f.add_subplot(2,2,1)
  ax.imshow(convert_batch_to_image_grid(train_batch[0].numpy()),
            interpolation='nearest')
  ax.set_title('training data originals')
  plt.axis('off')

  ax = f.add_subplot(2,2,2)
  ax.imshow(convert_batch_to_image_grid(train_reconstructions),
            interpolation='nearest')
  ax.set_title('training data reconstructions')
  plt.axis('off')

  ax = f.add_subplot(2,2,3)
  ax.imshow(convert_batch_to_image_grid(valid_batch[0].numpy()),
            interpolation='nearest')
  ax.set_title('validation data originals')
  plt.axis('off')

  ax = f.add_subplot(2,2,4)
  ax.imshow(convert_batch_to_image_grid(valid_reconstructions),
            interpolation='nearest')
  ax.set_title('validation data reconstructions')
  plt.axis('off')

  plt.show()


def main_test():
  # Set hyper-parameters.
  batch_size = 32
  image_size = 32

  # dataset
  transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (1, 1, 1))])

  trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)

  train_data_variance = np.var(trainset.data / 255.0)
  print(train_data_variance)

  train_dataset = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True, num_workers=1)

  testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
  test_dataset = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                         shuffle=False, num_workers=1)

  # classes = ('plane', 'car', 'bird', 'cat',
  #          'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

  def imshow(img):
    img = img / 1 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

  # Reconstructions
  train_batch = next(iter(train_dataset))
  valid_batch = next(iter(test_dataset))
  
  def convert_batch_to_image_grid_tensorflow(image_batch):
    reshaped = (image_batch.reshape(4, 8, 32, 32, 3)
                .transpose(0, 2, 1, 3, 4)
                .reshape(4 * 32, 8 * 32, 3))
    return reshaped + 0.5
  # pytorch
  def convert_batch_to_image_grid(image_batch):
    reshaped = (image_batch.transpose(0, 2, 3, 1)
                .reshape(4, 8, 32, 32, 3)
                .transpose(0, 2, 1, 3, 4)
                .reshape(4 * 32, 8 * 32, 3))
    return reshaped + 0.5

  f = plt.figure(figsize=(16,8))
  ax = f.add_subplot(2,2,1)
  ax.imshow(convert_batch_to_image_grid(train_batch[0].numpy()),
            interpolation='nearest')
  ax.set_title('training data originals')
  plt.axis('off')

  
  # ax = f.add_subplot(2,2,2)
  # ax.imshow(convert_batch_to_image_grid(train_reconstructions),
  #           interpolation='nearest')
  # ax.set_title('training data reconstructions')
  # plt.axis('off')

  # ax = f.add_subplot(2,2,3)
  # ax.imshow(convert_batch_to_image_grid(valid_batch['images'].numpy()),
  #           interpolation='nearest')
  # ax.set_title('validation data originals')
  # plt.axis('off')

  # ax = f.add_subplot(2,2,4)
  # ax.imshow(convert_batch_to_image_grid(valid_reconstructions),
  #           interpolation='nearest')
  # ax.set_title('validation data reconstructions')
  # plt.axis('off')
  plt.show()

  #for step_index, data in enumerate(train_dataset):
  # dataiter = iter(train_dataset)
  # data = dataiter.next()
  # input = data[0]
  # input = input.transpose(1,3)
  # print(input)
  # output = encoder(data[0])
  # output = pre_vq_conv1(output)
  # output = decoder(output)
  # print(output)



if __name__ == '__main__':

  input = torch.ones(3,32,32)
  # transor = transforms.Normalize((0.5, 0.5, 0.5), (1, 1, 1))
  # output = transor(input)
  # print(output)
  
  #main()
  # main_test()
  


