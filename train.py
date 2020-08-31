import time
import datetime
import pytz  
import argparse
import numpy as np
import torch 
import torch.nn as nn
import torch.nn.functional as F
import torchvision.utils as vutils


from model.net import MonocularDepthModel 
from model.loss import LossNetwork, combined_loss, mean_l2_loss
from model.metrics import evaluate_predictions
from model.dataloader import DataLoaders
from utils import *
from evaluate import infer_depth, evaluate_list

class Trainer():
  def __init__(self, data_path):
    self.dataloaders = DataLoaders(data_path)  

  def train_and_evaluate(self, config):
    batch_size = config['batch_size']
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    train_dataloader = self.dataloaders.get_train_dataloader(batch_size = batch_size) 
    num_batches = len(train_dataloader)

    model = MonocularDepthModel()
    model = model.to(device)
    params = [param for param in model.parameters() if param.requires_grad == True]
    optimizer = torch.optim.Adam(params, config['lr'])
    
    loss_model = LossNetwork().to(device)
          
    if config['checkpoint']:
      load_checkpoint(config['checkpoint'], model, optimizer)
    
    print('Training...')  

    for epoch in range(config['epochs']):

      accumulated_per_pixel_loss = RunningAverage()
      accumulated_feature_loss = RunningAverage()
      accumulated_iteration_time = RunningAverage()
      epoch_start_time = time.time()

      for iteration, batch in enumerate(train_dataloader):
        model.train() 
        time_start = time.time()        

        optimizer.zero_grad()
        images, depths = batch['img'], batch['depth']
        images = normalize_batch(torch.autograd.Variable(images.to(device)))
        depths = torch.autograd.Variable(depths.to(device))

        predictions = model(images)

        predictions_normalized = normalize_batch(predictions)
        depths_normalized = normalize_batch(depths)

        feature_losses_predictions = loss_model(predictions_normalized)
        feature_losses_depths = loss_model(depths_normalized)

        per_pixel_loss = combined_loss(predictions, depths)
        accumulated_per_pixel_loss.update(per_pixel_loss, images.shape[0])

        feature_loss = mean_l2_loss(feature_losses_predictions.res3, feature_losses_depths.res3)
        accumulated_feature_loss.update(feature_loss, images.shape[0])

        total_loss = per_pixel_loss + feature_loss
        total_loss.backward()
        optimizer.step()

        time_end = time.time()
        accumulated_iteration_time.update(time_end - time_start)
        eta = str(datetime.timedelta(seconds = int(accumulated_iteration_time() * (num_batches - iteration))))

        if iteration % config['log_interval'] == 0: 
          print(datetime.datetime.now(pytz.timezone('Asia/Kolkata')), end = ': ')
          print('Epoch: %d [%d / %d] ; it_time: %f (%f) ; eta: %s' % (epoch, iteration, num_batches, time_end - time_start, accumulated_iteration_time(), eta))
          print('Average per-pixel loss: %f; Average feature loss: %f' % (accumulated_per_pixel_loss(), accumulated_feature_loss()))
                              
      epoch_end_time = time.time()
      print('Epoch %d complete, time taken: %s' % (epoch, str(datetime.timedelta(seconds = int(epoch_end_time - epoch_start_time)))))
      torch.cuda.empty_cache()

      save_checkpoint({
                  'epoch': epoch,
                  'state_dict': model.state_dict(), 	
                  'optim_dict': optimizer.state_dict()}, config['checkpoint_dir'])

      print('Epoch %d saved\n\n' % (epoch))

if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='Training of depth estimation model')
  '''REQUIRED ARGUMENTS'''
  parser.add_argument('--data_dir', help='Train directory path - should contain the \'data\' folder', required = True)
  parser.add_argument('--batch_size', type=int, help='Batch size to process the train data', required = True)
  parser.add_argument('--checkpoint_dir', help='Directory to save checkpoints in', required = True)
  parser.add_argument('--epochs', type = int, help = 'Number of epochs', required = True)
  '''OPTIONAL ARGUMENTS'''
  parser.add_argument('--checkpoint', help='Model checkpoint path', default = None)
  parser.add_argument('--lr', help = 'Learning rate', default = 3e-4)
  parser.add_argument('--log_interval', help = 'Interval to print the avg. loss and metrics', default = 50)

  args = parser.parse_args()

  if not os.path.isdir(args.checkpoint_dir):
    os.mkdir(args.checkpoint_dir)

  trainer = Trainer(args.data_dir)
  trainer.train_and_evaluate(vars(args))
