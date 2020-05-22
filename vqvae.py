import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as numpy

class ResidualBlock(nn.Module):
  def __init__(self, num_hiddens,
    num_residual_hiddens):
    super(ResidualBlock,self).__init__()
    self._num_hiddens = num_hiddens
    self._num_residual_hiddens = num_residual_hiddens

    self.conv3 = nn.Conv2d(3,num_residual_hiddens,3,padding=1)
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

    self._layers = []
    for _ in range(num_residual_layers):
      conv3 = nn.Conv2d(3,num_residual_hiddens,3,padding=1)
      conv1 = nn.Conv2d(num_residual_hiddens,num_hiddens,1)
      self._layers.append((conv3,conv1))

  def forward(self, inputs):
    h = inputs
    for conv3, conv1 in self._layers:
      conv3_out = conv3(F.relu(h))
      conv1_out = conv1(F.relu(conv3_out))
      h = h + conv1_out
    return F.relu(h)

class Encoder(nn.Modele):
  def __init__(self, num_hiddens, num_residual_layers, num_residual_hiddens):
    super(Encoder, self).__init__()
    