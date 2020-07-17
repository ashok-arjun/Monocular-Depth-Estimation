import numpy as np
import torch 

from model.net import evaluate_predictions, combined_loss
from utils import RunningAverage

from torch.nn import Upsample

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
  2. Compare validation set depths with bilinearly upsampled output depths, calculate loss and metrics
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

def evaluate_list(model, samples, crop, batch_size):
  """
  Evaluates on the test data(with the eigen crop)
  """
  device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
  model = model.to(device)
  model.eval()
  upsample_2x = Upsample(scale_factor = 2, mode = 'bilinear', align_corners = True) #TODO: try without aligning the corners

  all_predictions = []
  all_depths = []

  with torch.no_grad():

    for i in range(0,len(samples),batch_size):
      batch_samples = samples[i:i+batch_size]
      images = torch.stack([bs['img'] for bs in batch_samples])
      depths = torch.stack([bs['depth'] for bs in batch_samples])

      images = torch.autograd.Variable(images.to(device))
      depths = torch.autograd.Variable(depths.to(device))

      # TODO: try learning the upsampling(upconv)
      # TODO: try averaging mirror prediction's mirror and current prediction
      
      # without mirroring
      predictions_unflipped = upsample_2x(model(images))
      predictions_unflipped = predictions_unflipped[:, :, crop[0]:crop[1]+1, crop[2]:crop[3]+1]

      #mirroring
      predictions_flipped = upsample_2x(model(torch.from_numpy(images.cpu().numpy()[:,:,:,::-1].copy()).to(device)))
      predictions_from_flipped = torch.from_numpy(predictions_flipped.cpu().numpy()[:,:,:,::-1].copy()).to(device)
      predictions_from_flipped = predictions_from_flipped[:, :, crop[0]:crop[1]+1, crop[2]:crop[3]+1]

      # averaging them
      predictions = 0.5 * predictions_unflipped + 0.5 * predictions_from_flipped

      # eigen crop depth
      depths = depths[:, :, crop[0]:crop[1]+1, crop[2]:crop[3]+1]

      all_predictions.append(predictions)
      all_depths.append(depths)
    # END FOR

    all_predictions = torch.stack(all_predictions)
    all_depths = torch.stack(all_depths)

    # loss = combined_loss(all_predictions, all_depths)
    metrics = evaluate_predictions(all_predictions, all_depths)

    return metrics


