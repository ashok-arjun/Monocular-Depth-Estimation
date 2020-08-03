import time
import datetime
import pytz  

import numpy as np
import torch 
import torch.nn as nn
import torch.nn.functional as F
import torchvision.utils as vutils # contains useful functions like make_grid
from torch.utils.tensorboard import SummaryWriter


from model.net import DenseDepth, DenseDepthWithUpconvolution, evaluate_predictions, combined_loss 
from model.dataloader import DataLoaders
from utils import *
from evaluate import evaluate

# shift these to config files or inside the class later
DATA_PATH = 'nyu_data.zip'

class Trainer():
  def __init__(self, data_path = DATA_PATH, resized = True):
    print('Loading data...')
    self.dataloaders = DataLoaders(path = data_path, resized = resized)  
    self.resized = resized
    print('Data loaded!')

  def train_and_evaluate(self, config, checkpoint_file, local):
    """
    TODO: log other values/images
    """
    batch_size = config['batch_size']
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    train_dataloader = self.dataloaders.get_train_dataloader(batch_size = batch_size) 
    num_batches = len(train_dataloader)

    model = DenseDepth()
    if self.resized == False:
      model = DenseDepthWithUpconvolution(model)
    model = model.to(device)
    params = [param for param in model.parameters() if param.requires_grad == True]
    print('A total of %d parameters in present model' % (len(params)))
    optimizer = torch.optim.Adam(params, config['lr'])
    
    best_rmse = 9e20
    is_best = False
    best_test_rmse = 9e20 
    
    if local:
      print('Loading checkpoint from local storage:',checkpoint_file)
      load_checkpoint(checkpoint_file, model, optimizer)
      print('Loaded checkpoint from local storage:',checkpoint_file)    

    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size = config['lr_scheduler_step_size'], gamma = 0.1)
    for i in range(config['start_epoch']):
      lr_scheduler.step() # step the scheduler for the already done epochs
    print('Training...')    
    model.train()  
  
    for epoch in range(config['start_epoch'], config['epochs']):
      
      accumulated_loss = RunningAverage()
      accumulated_iteration_time = RunningAverage()
      epoch_start_time = time.time()

      for iteration, batch in enumerate(train_dataloader):

        time_start = time.time()        

        optimizer.zero_grad()
        images, depths = batch['img'], batch['depth']
        images = torch.autograd.Variable(images.to(device))
        depths = torch.autograd.Variable(depths.to(device))

        # depths = 1000.0/depths

        predictions = model(images)

        loss = combined_loss(predictions, depths)
        accumulated_loss.update(loss, images.shape[0])

        loss.backward()
        optimizer.step()

        time_end = time.time()
        accumulated_iteration_time.update(time_end - time_start)
        eta = str(datetime.timedelta(seconds = int(accumulated_iteration_time() * (num_batches - iteration))))

        if iteration % config['training_loss_log_interval'] == 0: 
          print(datetime.datetime.now(pytz.timezone('Asia/Kolkata')), end = ' ')
          print('At epoch %d[%d/%d]' % (epoch, iteration, num_batches))

        if iteration % config['other_metrics_log_interval'] == 0:

          print('Epoch: %d [%d / %d] ; it_time: %f (%f) ; eta: %s ; loss: %f (%f)' % (epoch, iteration, num_batches, time_end - time_start, accumulated_iteration_time(), eta, loss.item(), accumulated_loss()))
          metrics = evaluate_predictions(predictions, depths)

          test_images, test_depths, test_preds, test_loss, test_metrics = evaluate(model, self.dataloaders.get_val_dataloader, batch_size = config['test_batch_size']) ; model.train() # evaluate(in model.eval()) and back to train

          if metrics['rmse'] < best_rmse: 
            best_rmse = metrics['rmse']
            is_best = True

          save_checkpoint({'iteration': iteration, 
                          'state_dict': model.state_dict(), 
                          'optim_dict': optimizer.state_dict()},
                          is_best = is_best,
                          checkpoint_dir = 'experiments/', train = True)

          if test_metrics['rmse'] < best_test_rmse:
            best_test_rmse = test_metrics['rmse']

          is_best = False
          
          del test_images
          del test_depths
          del test_preds
          
        del predictions
        del depths
        del images 
 
        
                               

      epoch_end_time = time.time()
      print('Epoch %d complete, time taken: %s' % (epoch, str(datetime.timedelta(seconds = int(epoch_end_time - epoch_start_time)))))
      lr_scheduler.step() 
      torch.cuda.empty_cache()
