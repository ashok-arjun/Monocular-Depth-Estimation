import numpy as np
import torch 

from model.net import evaluate_predictions, combined_loss
from utils import RunningAverage

import time
import datetime

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
 

def evaluate_full(model, dataloader_getter, config):
  """
  Evaluates for one complete epoch, averages and returns the loss and metrics
  Two ways:
  1. Compare the validation set with the downsampled outputs, and calculate loss, metrics 
  2. Compare validation set depths with bilinearly upsampled output depths, calculate loss and metricsz
  """ 
  batch_size = config['test_batch_size']
  print_every = config['test_metrics_log_interval']

  device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
  model = model.to(device)
  
  model.eval()

  test_dl = dataloader_getter(batch_size = batch_size, shuffle = True) 
  num_iterations = len(test_dl)
  print('Batch size: %d, Number of batches: %d' % (batch_size, num_iterations) )

  avg_loss = RunningAverage()
  avg_metrics = AverageMetrics()

  start_time = time.time()
  with torch.no_grad():
    for iteration, batch in enumerate(test_dl):
      images, depths = batch['img'], batch['depth']
      images = torch.autograd.Variable(images.to(device))
      depths = torch.autograd.Variable(depths.to(device))

      predictions = model(images)

      loss = combined_loss(predictions, depths)
      avg_loss.update(loss)

      metrics = evaluate_predictions(predictions, depths)
      avg_metrics.update(metrics)

      if iteration % print_every == 0:
        print('Iteration [%d/%d], Average loss: %f \n Average metrics' % (iteration, num_iterations, avg_loss()), end = ' ')
        print(avg_metrics())  

  end_time = time.time()
  print('Time taken: %s' % (str(datetime.timedelta(seconds = end_time - start_time))))
  print('Average loss: %f \nAverage metrics:' % (avg_loss()))
  print(avg_metrics())
  return avg_loss(), avg_metrics() 

def evaluate(model, dataloader_getter, batch_size):
  """
  Evaluates on a single iteration of the dataloader, with batch_size images
  :returns the loss value and metrics dict
  """
  device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
  model = model.to(device)
  
  model.eval()

  test_dl = dataloader_getter(batch_size = batch_size, shuffle = True) 

  sample = next(iter(test_dl))
  images, depths = sample['img'], sample['depth']
  images = torch.autograd.Variable(images.to(device))
  depths = torch.autograd.Variable(depths.to(device))
  
  with torch.no_grad():
    predictions = model(images)
    loss = combined_loss(predictions, depths)
    metrics = evaluate_predictions(predictions, depths)

  return images, depths, predictions, loss, metrics




