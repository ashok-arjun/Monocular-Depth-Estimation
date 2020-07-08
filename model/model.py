"""
Model, loss and metrics
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

"""
MODEL
"""

# inherit nn.Module with __init__(self, params) and forward(self, img)
# params can contain dropout rate, batch_norm or group_norm, number of channels in each layer, etc..
# try out Densenet 121(7 million parameters) & then other models(if needed)
# return number of channels in each bridge layer for the decoder's 4 upsampling blocks
# bridge layer should inherit from nn.Sequential, it should perform upsampling to 2nd input's size, concat their dimensions and apply
# conv-relu-conv-relu

class Bridge(nn.Sequential):
  def __init__(self, in_channels, out_channels):
    self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size = 3, padding = 1)
    self.act1 = nn.LeakyReLU(0.2) # tune 
    self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size = 3, padding = 1)
    self.act2 = nn.LeakyReLU(0.2) # tune 

  def forward(self, encoder_feature_map, decoder_feature_map):
    upsampled_decoder_map = self.bilinear_upsampling(decoder_feature_map, *encoder_feature_map.shape)
    concatenated_maps = torch.cat((upsampled_decoder_map, encoder_feature_map), dim = 1)
    return self.act2(self.conv2(self.act1(self.conv1(concatenated_maps))))   

  def bilinear_upsampling(x, shape):
    return F.interpolate(x, size = (shape[2], shape[3]), mode='bilinear', align_corners = True)  

class Encoder(nn.Module):
  def __init__(self):
    super(Encoder, self).__init__()
    self.backbone = torchvision.models.densenet121(pretrained = True)

  def forward(self, images):
    """
    Reference: the keys  of modules in densenet121 are 
    ['conv0', 'norm0', 'relu0', 'pool0', 'denseblock1', 'transition1', 
    'denseblock2', 'transition2', 'denseblock3', 'transition3', 'denseblock4', 'norm5']
    -----------------12 modules in densenet121, + images = 13 modules in total----------------
    """
    feature_maps = [images]
    
    for module in self.backbone.features._modules.values(): 
      feature_maps.append(module(feature_maps[-1]))

    return feature_maps  

class Decoder(nn.Module):
  def __init__(self, final_channels = 1024, delta = 0.5):
    """
    final_channels represents the number of channels from the last dense block
    delta represents the decay in the number of channels to operate the decoder at
    """
    super(Decoder, self).__init__()

    self.final_channels = int(final_channels * delta)

    self.conv6 = nn.Conv2d(1024,self.final_channels, kernel_size=1, stride=1, padding=0)   # named conv 6, as the final batch normalisation layer is bn5

    self.bridge1 = Bridge(in_channels = 256 + self.final_channels, out_channels = self.final_channels // 2) 
    self.bridge2 = Bridge(in_channels = 128 + self.final_channels // 2, out_channels = self.final_channels // 4)  
    self.bridge3 = Bridge(in_channels = 64 + self.final_channels // 4, out_channels = self.final_channels // 8)  
    self.bridge4 = Bridge(in_channels = 64 + self.final_channels // 8, out_channels = self.final_channels // 16) 
    # self.bridge5 = Bridge(, out_channels = self.final_channels // 32) # requires depth size == image size

    self.conv7 = nn.Conv2d(self.final_channels // 16, 1, kernel_size=1, stride=1, padding=0) # convert to one channel(depth)  

  def forward(self, encoder_feature_maps):
    """
    Gets intermediate feature maps from the encoder, and applies bridge connections to the decoder layers
    """

    conv0, pool0, transition1, transition2, dense4 = encoder_feature_maps[3], encoder_feature_maps[4], encoder_feature_maps[6], encoder_feature_maps[8], encoder_feature_maps[11]
    
    conv6_map = self.conv6(dense4)
    bridge1_map = self.bridge1(transition2, conv6_map) 
    bridge2_map = self.bridge1(transition1, bridge1_map) 
    bridge3_map = self.bridge1(pool0, bridge2_map) 
    bridge4_map = self.bridge1(conv0, bridge3_map)
    depth_map = self.conv7(bridge4_map)
    return depth_map



class DenseDepth(nn.Module):
  def __init__(self):
    super(DenseDepth, self).__init__()
    self.encoder = Encoder()
    self.decoder = Decoder()  

  def forward(self, images):
    return self.decoder(self.encoder(images))  
"""
METRICS
"""

def evaluate_predictions(predictions, truth):
  """
  Defines 5 metrics used to evaluate the depth estimation models
  """
  ratio_max = torch.max((predictions/truth),(truth/predictions)) #element wise maximum
  print(ratio_max.shape)
  d1 = torch.mean((ratio_max < 1.25     ).float()) # mean of the bool tensor
  d2 = torch.mean((ratio_max < 1.25 ** 2).float()) # mean of the bool tensor
  d3 = torch.mean((ratio_max < 1.25 ** 3).float()) # mean of the bool tensor
  relative_error = torch.mean(torch.abs(predictions - truth)/truth)
  rmse = torch.sqrt(mean_l2_loss(predictions, truth))
  log10_error = torch.mean(torch.abs(torch.log10(predictions) - torch.log10(truth)))
  return d1, d2, d3, relative_error, rmse, log10_error

"""
LOSSES
"""
def mean_l1_loss(predictions, truth):
  loss = nn.L1Loss(reduction = 'mean')
  return loss(predictions, truth)

def mean_l2_loss(predictions, truth):
  loss = nn.MSELoss(reduction = 'mean') 
  return loss(predictions, truth) 

def mean_l1_log_loss(predictions, truth):
  """
  L1 loss with log
  """
  loss = torch.mean(torch.abs(torch.log(predictions) - torch.log(truth)))
  return loss

def berHu_loss(predictions, truth):
  """
  Obtained from Laina et. al.
  predictions and truth are batched
  """
#  TODO: check with other implementation

  l1_error = torch.abs(predictions - truth) # per pixel 
  
  c = 0.2 * torch.max(l1_error) # maximum l1 loss across batch across pixels, check this line for errors(CPU - GPU)
  
  if c == 0:
    return torch.mean(l1_error)
  
  berhu_part_1 = -F.threshold(-l1_error, -c, 0)
  berhu_part_2 = (F.threshold(l1_error, c, 0) ** 2 + (c ** 2)) / (2 * c) # check implementation

  return torch.sum(berhu_part_1 + berhu_part_2)

# TODO: 
# Gradient loss(https://discuss.pytorch.org/t/how-to-calculate-the-gradient-of-images/1407/6?u=arjunashok)
# SSIM loss(https://github.com/Po-Hsun-Su/pytorch-ssim/blob/master/pytorch_ssim/__init__.py)
