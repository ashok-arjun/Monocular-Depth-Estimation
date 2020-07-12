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

# inherit nn.Module with __init__(self, params) and forward(self, img)
# params can contain dropout rate, batch_norm or group_norm, number of channels in each layer, etc..
# try out Densenet 121(7 million parameters) & then other models(if needed)
# return number of channels in each bridge layer for the decoder's 4 upsampling blocks
# bridge layer should inherit from nn.Sequential, it should perform upsampling to 2nd input's size, concat their dimensions and apply
# conv-relu-conv-relu

class Bridge(nn.Sequential):
  def __init__(self, in_channels, out_channels):
    super(Bridge, self).__init__()
    self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size = 3, padding = 1)
    self.act1 = nn.LeakyReLU(0.2) # tune 
    self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size = 3, padding = 1)
    self.act2 = nn.LeakyReLU(0.2) # tune 

  def forward(self, encoder_feature_map, decoder_feature_map):
    upsampled_decoder_map = self.bilinear_upsampling(decoder_feature_map, encoder_feature_map.shape)
    concatenated_maps = torch.cat((upsampled_decoder_map, encoder_feature_map), dim = 1)
    return self.act2(self.conv2(self.act1(self.conv1(concatenated_maps))))   

  def bilinear_upsampling(self, x, shape):
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
    bridge2_map = self.bridge2(transition1, bridge1_map) 
    bridge3_map = self.bridge3(pool0, bridge2_map) 
    bridge4_map = self.bridge4(conv0, bridge3_map)
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
  d1 = torch.mean((ratio_max < 1.25     ).float()) # mean of the bool tensor
  d2 = torch.mean((ratio_max < 1.25 ** 2).float()) # mean of the bool tensor
  d3 = torch.mean((ratio_max < 1.25 ** 3).float()) # mean of the bool tensor
  relative_error = torch.mean(torch.abs(predictions - truth)/truth)
  rmse = torch.sqrt(mean_l2_loss(predictions, truth))
  log10_error = torch.mean(torch.abs(torch.log10(predictions) - torch.log10(truth)))
  return {'d1_accuracy':d1, 'd2_accuracy':d2, 'd3_accuracy':d3, 'relative_err':relative_error, 'rmse':rmse, 'log10_error':log10_error}

"""
LOSSES
"""


"""
Combined loss
"""

def combined_loss(predictions, truth):
  return 0.1 * mean_l1_loss(predictions, truth) + gradient_loss(predictions, truth)  + (1 - ssim(predictions, truth))*0.5
"""
Separate loss function
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
  
  c = 0.2 * torch.max(l1_error).data.cpu().numpy() # maximum l1 loss across batch across pixels, check this line for errors(CPU - GPU)

  if c == 0:
    return torch.mean(l1_error)
  
  berhu_part_1 = -F.threshold(-l1_error, -c, 0)
  berhu_part_2 = (F.threshold(l1_error, c, 0) ** 2 + (c ** 2)) / (2 * c) # check implementation

  return torch.sum(berhu_part_1 + berhu_part_2)


# taken from https://discuss.pytorch.org/t/how-to-calculate-the-gradient-of-images/1407/6
def gradient_loss(predictions, truth, alpha=1):

    def gradient(x):
        # idea from tf.image.image_gradients(image)
        # https://github.com/tensorflow/tensorflow/blob/r2.1/tensorflow/python/ops/image_ops_impl.py#L3441-L3512
        # x: (b,c,h,w), float32 or float64
        # dx, dy: (b,c,h,w)

        h_x = x.size()[-2]
        w_x = x.size()[-1]
        # gradient step=1
        left = x
        right = F.pad(x, [0, 1, 0, 0])[:, :, :, 1:]
        top = x
        bottom = F.pad(x, [0, 0, 0, 1])[:, :, 1:, :]

        # dx, dy = torch.abs(right - left), torch.abs(bottom - top)
        dx, dy = right - left, bottom - top 
        # dx will always have zeros in the last column, right-left
        # dy will always have zeros in the last row,    bottom-top
        dx[:, :, :, -1] = 0
        dy[:, :, -1, :] = 0

        return dx, dy

    # gradient
    gen_dx, gen_dy = gradient(predictions)
    gt_dx, gt_dy = gradient(truth)
    #
    grad_diff_x = torch.abs(gt_dx - gen_dx)
    grad_diff_y = torch.abs(gt_dy - gen_dy)

    # condense into one tensor and avg
    return torch.mean(grad_diff_x ** alpha + grad_diff_y ** alpha)


"""
SSIM loss
"""
def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
    return gauss/gauss.sum()

def create_window(window_size, channel=1):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    return window

def ssim(img1, img2, val_range = 1000.0, window_size=11, window=None, size_average=True, full=False):
    L = val_range

    padd = 0
    (_, channel, height, width) = img1.size()
    if window is None:
        real_size = min(window_size, height, width)
        window = create_window(real_size, channel=channel).to(img1.device)

    mu1 = F.conv2d(img1, window, padding=padd, groups=channel)
    mu2 = F.conv2d(img2, window, padding=padd, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=padd, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=padd, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=padd, groups=channel) - mu1_mu2

    C1 = (0.01 * L) ** 2
    C2 = (0.03 * L) ** 2

    v1 = 2.0 * sigma12 + C2
    v2 = sigma1_sq + sigma2_sq + C2
    cs = torch.mean(v1 / v2)  # contrast sensitivity

    ssim_map = ((2 * mu1_mu2 + C1) * v1) / ((mu1_sq + mu2_sq + C1) * v2)

    if size_average:
        ret = ssim_map.mean()
    else:
        ret = ssim_map.mean(1).mean(1).mean(1)

    if full:
        return ret, cs

    return ret
