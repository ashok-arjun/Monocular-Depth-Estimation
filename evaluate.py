import numpy as np
import torch 
import time
import datetime
import argparse

from torch.nn import Upsample
from model.net import MonocularDepthModel
from model.dataloader import NYUDepthTestDataset, get_test_transforms
from model.metrics import evaluate_predictions
from model.loss import combined_loss
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

def infer_depth(image_tensor, model, upsample = True):
  '''Image_tensor should be of shape C * H * W (and between 0 and 1) and H,W should be divisible by 32 perfectly.
  If true depth is provided, it should also be a tensor(resized to the model prediction size)'''
  device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
  model = model.to(device)
  model.eval()

  image = normalize_batch(torch.autograd.Variable(image_tensor.unsqueeze(0).to(device)))
 
  with torch.no_grad():
    depth = model(image)
  if upsample:
    depth = Upsample(scale_factor = 2, mode = 'bilinear', align_corners = True)(depth)
  return depth

def evaluate(model, test_dataloader, batch_size, model_upsample=True):
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

    for it, batch in enumerate(test_dataloader):
      # check this
      
      images = batch[0]['img']
      depths = batch[0]['depth']
      crop = batch[1]
      
      images = torch.autograd.Variable(images.to(device))
      depths = torch.autograd.Variable(depths.to(device))

      images = normalize_batch(images)
      
      if model_upsample:
        predictions_unflipped = upsample_2x(model(images)) 
      else:
        predictions_unflipped = model(images) 
      predictions_unflipped = predictions_unflipped[:, :, crop[0]:crop[1]+1, crop[2]:crop[3]+1]


      if model_upsample:
        predictions_flipped = upsample_2x(model(torch.from_numpy(images.cpu().numpy()[:,:,:,::-1].copy()).to(device))) 
      else:
        predictions_flipped = model(torch.from_numpy(images.cpu().numpy()[:,:,:,::-1].copy()).to(device)) 

      predictions_from_flipped = torch.from_numpy(predictions_flipped.cpu().numpy()[:,:,:,::-1].copy()).to(device)
      predictions_from_flipped = predictions_from_flipped[:, :, crop[0]:crop[1]+1, crop[2]:crop[3]+1]

      predictions = 0.5 * predictions_unflipped + 0.5 * predictions_from_flipped

      depths = depths[:, :, crop[0]:crop[1]+1, crop[2]:crop[3]+1]
      all_predictions.append(predictions)
      all_depths.append(depths)
    # END FOR

    all_predictions = torch.cat(all_predictions, dim = 0)
    all_depths = torch.cat(all_depths, dim = 0)

    metrics = evaluate_predictions(all_predictions, all_depths)

    return metrics

if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='Evaluation of depth estimation model on either test data/own images')
  parser.add_argument('--model', help='Model checkpoint path', required=True)
  parser.add_argument('--data_dir', help='Test data directory(If evaluation on test data)')
  parser.add_argument('--img', help='Image path(If evaluation on a single image)')
  parser.add_argument('--batch_size', type=int, help='Batch size to process the test data', default = 6)
  parser.add_argument('--output_dir', help='Directory to save output depth images', default = 'outputs')

  args = parser.parse_args()

  if (args.data is None and args.img is None) or (args.data and args.img):
    raise Exception('Please provide either the test .zip path or a single image\'s path')

  device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
  model = MonocularDepthModel().to(device) 
  load_checkpoint(args.model, model)  

  if args.data_dir:
    print('Evaluating on test data...')
    dataset = NYUDepthTestDataset(args.data_dir, get_test_transforms())
    dataloader = torch.data.utils.Dataloader(dataset, batch_size = args.batch_size, shuffle = False, num_workers = 0)
    test_metrics = evaluate_list(model, dataloader, args.batch_size, model_upsample = True)
    for key, value in test_metrics.items():	
      print('Test %s: %f' % (key, value))
  elif args.img:
    print('Evaluating on a single image...')
    # convert image to PIL image and then to Tensor
    # pass to infer_depth
    # convert back, and colormap apply
    if not os.path.isdir(args.output_dir):
      os.mkdir(args.output_dir)
    # matplotlib save
