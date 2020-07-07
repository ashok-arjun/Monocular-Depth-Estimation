"""
Model, loss and metrics
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

"""
MODEL
"""



"""
METRICS
"""

def evaluate_predictions(predictions, truth):
  """
  Defines 5 metrics used to evaluate the depth estimation models
  """
  # TODO: check for batch_size > 1(does it require summing) 
  # TODO: check all errors
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
  TODO: check with other implementation
  """

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
