"""
Model, loss and metrics
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from math import exp

"""
MODEL
"""
class Bridge(nn.Sequential):
  def __init__(self, in_channels, out_channels):
    super(Bridge, self).__init__()
    self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size = 3, padding = 1)
    self.act1 = nn.LeakyReLU(0.2)  
    self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size = 3, padding = 1)
    self.act2 = nn.LeakyReLU(0.2)  

  def forward(self, encoder_feature_map, decoder_feature_map):
    upsampled_decoder_map = self.bilinear_upsampling(decoder_feature_map, encoder_feature_map.shape)
    concatenated_maps = torch.cat((upsampled_decoder_map, encoder_feature_map), dim = 1)
    return self.act2(self.conv2(self.act1(self.conv1(concatenated_maps))))   

  def bilinear_upsampling(self, x, shape):
    return F.interpolate(x, size = (shape[2], shape[3]), mode='bilinear', align_corners = True)  

class Encoder(nn.Module):
  def __init__(self, backbone = 'densenet121'):
    super(Encoder, self).__init__()
    if backbone == 'densenet121':
      self.backbone = torchvision.models.densenet121(pretrained = True, progress = False)
    elif backbone == 'densenet161':
      self.backbone = torchvision.models.densenet161(pretrained = True, progress = False)

  def forward(self, images):
    feature_maps = [images]
    
    for module in self.backbone.features._modules.values(): 
      feature_maps.append(module(feature_maps[-1]))

    return feature_maps  

class Decoder(nn.Module):
  def __init__(self, backbone = 'densenet121', delta = 0.5):
    """
    backbone: densenet121 or densenet161
    final_channels represents the number of channels from the last dense block
    delta represents the decay in the number of channels to operate the decoder at
    """
    super(Decoder, self).__init__()
    
    if backbone is 'densenet121':
      final_channels = 1024 
      backbone_in_channels = [256, 128, 64, 64]
    elif backbone is 'densenet161':
      final_channels = 2208 
      backbone_in_channels = [384, 192, 96, 96]
      
    self.final_channels = int(final_channels * delta)

    self.conv6 = nn.Conv2d(final_channels, self.final_channels, kernel_size=1, stride=1, padding=0)   

    self.bridge1 = Bridge(in_channels = backbone_in_channels[0] + self.final_channels, out_channels = self.final_channels // 2) 
    self.bridge2 = Bridge(in_channels = backbone_in_channels[1] + self.final_channels // 2, out_channels = self.final_channels // 4)  
    self.bridge3 = Bridge(in_channels = backbone_in_channels[2] + self.final_channels // 4, out_channels = self.final_channels // 8)  
    self.bridge4 = Bridge(in_channels = backbone_in_channels[3] + self.final_channels // 8, out_channels = self.final_channels // 16) 

    self.conv7 = nn.Conv2d(self.final_channels // 16, 1, kernel_size=1, stride=1, padding=0)   

  def forward(self, encoder_feature_maps):
    """
    Gets intermediate feature maps from the encoder, and applies bridge connections to the decoder layers
    """

    conv0, pool0, transition1, transition2, dense4 = encoder_feature_maps[3], encoder_feature_maps[4], encoder_feature_maps[6], encoder_feature_maps[8], encoder_feature_maps[11]
    
    conv6_map = self.conv6(dense4)
    bridge1_map = self.bridge1(transition2, conv6_map) 
    bridge2_map = self.bridge2(transition1, bridge1_map) 
    bridge3_map = self.bridge3(pool0, bridge2_map) 
    bridge4_map = self.bridge4(conv0, bridge3_map)
    depth_map = self.conv7(bridge4_map)
    return depth_map



class MonocularDepthModel(nn.Module):
  def __init__(self, backbone = 'densenet121'):
    super(MonocularDepthModel, self).__init__()
    if backbone != 'densenet121' and backbone != 'densenet161':
      raise Exception('Backbone %s is not supported' % (backbone))
    self.encoder = Encoder(backbone = backbone)
    self.decoder = Decoder(backbone = backbone)  

  def forward(self, images):
    return torch.sigmoid(self.decoder(self.encoder(images)))
