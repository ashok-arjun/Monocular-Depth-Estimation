import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from math import exp
from collections import namedtuple


class LossNetwork(torch.nn.Module):
  '''
  This has to receive a tensor between 0 and 1, normalized by Image mean and SD
  '''
  def __init__(self, requires_grad=False):
    super(LossNetwork, self).__init__()

    resnet50 = models.resnet50(pretrained = True)
    modules=list(resnet50.children())[:-1]
    resnet50=nn.Sequential(*modules)
    self.slice1 = torch.nn.Sequential()
    self.slice2 = torch.nn.Sequential()
    self.slice3 = torch.nn.Sequential()
    self.slice4 = torch.nn.Sequential()
    for x in range(0,5):
      self.slice1.add_module(str(x), resnet50[x])
    for x in range(5,6):
      self.slice2.add_module(str(x), resnet50[x])
    for x in range(6,7):
      self.slice3.add_module(str(x), resnet50[x])
    for x in range(7,8):
      self.slice4.add_module(str(x), resnet50[x])
    if not requires_grad:
      for param in self.parameters():
        param.requires_grad = False

  def forward(self, X):
    h = self.slice1(X)
    h_res_1 = h
    h = self.slice2(h)
    h_res_2 = h
    h = self.slice3(h)
    h_res_3 = h
    h = self.slice4(h)
    h_res_4 = h
    outputs = namedtuple("ResidualOutputs", ['res1', 'res2', 'res3', 'res4'])
    out = outputs(h_res_1, h_res_2, h_res_3, h_res_4)
    return out

"""
Combined loss
"""

def combined_loss(predictions, truth):
  loss = 0.1 * mean_l1_loss(predictions, truth) + gradient_loss(predictions, truth)  + (1 - ssim(predictions, truth)) * 0.5
  return loss

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

def ssim(img1, img2, val_range = 1.0, window_size=11, window=None, size_average=True, full=False):
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