import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as numpy

import collections

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

    self._enc_1 = nn.Conv2d(3, self._num_hiddens // 2, 4, 2)
    self._enc_2 = nn.Conv2d(self._num_hiddens // 2, self._num_hiddens, 4, 2)
    self._enc_3 = nn.Conv2d(self._num_hiddens, self._num_hiddens, 3, 1)
    
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


net = Encoder(10,2,12)
#net = ResidualStack(10,2,12)
print(net)
params = list(net.parameters())
print(len(params))
print(params[0].size())

