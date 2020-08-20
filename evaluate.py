import numpy as np
import torch 
import time
from model.metrics import evaluate_predictions
from model.loss import combined_loss
from utils import RunningAverage
import datetime

from torch.nn import Upsample
from utils import *

class AverageMetrics:
  def __init__(self):
    self.d1_accuracy = RunningAverage()
    self.d2_accuracy = RunningAverage()
    self.d3_accuracy = RunningAverage()
    self.relative_err = RunningAverage()
    self.rmse = RunningAverage()
    self.log10_error = RunningAverage()
  def update(self, metrics_object):
    self.d1_accuracy.update(metrics_object['d1_accuracy'])
    self.d2_accuracy.update(metrics_object['d2_accuracy'])
    self.d3_accuracy.update(metrics_object['d3_accuracy'])
    self.relative_err.update(metrics_object['relative_err'])
    self.rmse.update(metrics_object['rmse']) 
    self.log10_error.update(metrics_object['log10_error'])
  def __call__(self):
    return {
    'd1_accuracy':self.d1_accuracy(), 
    'd2_accuracy':self.d2_accuracy(), 
    'd3_accuracy':self.d3_accuracy(), 
    'relative_err':self.relative_err(), 
    'rmse':self.rmse(), 
    'log10_error':self.log10_error()}    

def evaluate(model, dataloader_getter, batch_size):
  """
  Evaluates on a SINGLE iteration of the dataloader, with batch_size images
  :returns the loss value and metrics dict
  """
  device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
  model = model.to(device)
  
  model.eval()

  test_dl = dataloader_getter(batch_size = batch_size, shuffle = True) 

  sample = next(iter(test_dl))
  images, depths = sample['img'], sample['depth']
  images_ = normalize_batch(torch.autograd.Variable(images.to(device)))
  depths = torch.autograd.Variable(depths.to(device))
  
  with torch.no_grad():
    predictions = model(images_)
    loss = combined_loss(predictions, depths)
    metrics = evaluate_predictions(predictions, depths)

  return images, depths, predictions, loss, metrics

def evaluate_list(model, samples, crop, batch_size, model_upsample=True):
  """
  Evaluates on the test data(with the eigen crop). Function created for easy execution of test data(available as a list)
  """
  device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
  model = model.to(device)
  model.eval()
  if model_upsample:
    upsample_2x = Upsample(scale_factor = 2, mode = 'bilinear', align_corners = True) 

  all_predictions = []
  all_depths = []

  with torch.no_grad():

    for i in range(0,len(samples),batch_size):
      batch_samples = samples[i:i+batch_size]
      images = torch.stack([bs['img'] for bs in batch_samples])
      depths = torch.stack([bs['depth'] for bs in batch_samples])

      images = torch.autograd.Variable(images.to(device))
      depths = torch.autograd.Variable(depths.to(device))

      images = normalize_batch(images)
      
      if model_upsample:
        predictions_unflipped = upsample_2x(model(images)) 
      else:
        predictions_unflipped = model(images) 
      predictions_unflipped = predictions_unflipped[:, :, crop[0]:crop[1]+1, crop[2]:crop[3]+1]


      if model_upsample:
        predictions_flipped = upsample_2x(model(torch.from_numpy(images.numpy()[:,:,:,::-1].copy()))) 
      else:
        predictions_flipped = model(torch.from_numpy(images.cpu().numpy()[:,:,:,::-1].copy()).to(device)) # Model 2

      predictions_from_flipped = torch.from_numpy(predictions_flipped.cpu().numpy()[:,:,:,::-1].copy()).to(device)
      predictions_from_flipped = predictions_from_flipped[:, :, crop[0]:crop[1]+1, crop[2]:crop[3]+1]

      predictions = 0.5 * predictions_unflipped + 0.5 * predictions_from_flipped

      depths = depths[:, :, crop[0]:crop[1]+1, crop[2]:crop[3]+1]
      all_predictions.append(predictions)
      all_depths.append(depths)
    # END FOR

    all_predictions = torch.stack(all_predictions); a_p_shape = all_predictions.shape
    all_predictions = all_predictions.view(a_p_shape[0] * a_p_shape[1], a_p_shape[2], a_p_shape[3], a_p_shape[4])
    all_depths = torch.stack(all_depths); a_d_shape = all_depths.shape
    all_depths = all_depths.view(a_d_shape[0] * a_d_shape[1], a_d_shape[2], a_d_shape[3], a_d_shape[4])

    metrics = evaluate_predictions(all_predictions, all_depths)

    return metrics