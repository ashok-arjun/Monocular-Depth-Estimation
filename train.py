import time
import datetime
import pytz  

import numpy as np
import torch 
import torch.nn as nn
import torch.nn.functional as F
import torchvision.utils as vutils # contains useful functions like make_grid
from torch.utils.tensorboard import SummaryWriter
import wandb


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
    
    if wandb.run.resumed or local:
      if local:
        print('Loading checkpoint from local storage:',checkpoint_file)
        load_checkpoint(checkpoint_file, model, optimizer)
        print('Loaded checkpoint from local storage:',checkpoint_file)
      else:  
        print('Loading checkpoint from cloud storage:',checkpoint_file)
        load_checkpoint(wandb.restore(checkpoint_file).name, model, optimizer)
        print('Loaded checkpoint from cloud storage:',checkpoint_file)
        best_rmse = wandb.run.summary["best_train_rmse"]
        best_test_rmse = wandb.run.summary["best_test_rmse"]

    

    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size = config['lr_scheduler_step_size'], gamma = 0.1)
    for i in range(config['start_epoch']):
      lr_scheduler.step() # step the scheduler for the already done epochs
    print('Training...')    
    model.train()  
    wandb.watch(model)
  
    wandb_step = config['start_epoch'] * num_batches -1 # set it to the number of iterations done

    for epoch in range(config['start_epoch'], config['epochs']):
      
      accumulated_loss = RunningAverage()
      accumulated_iteration_time = RunningAverage()
      epoch_start_time = time.time()

      for iteration, batch in enumerate(train_dataloader):

        wandb_step += 1

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
          wandb.log({'Training loss': loss.item()}, step = wandb_step)

        if iteration % config['other_metrics_log_interval'] == 0:

          print('Epoch: %d [%d / %d] ; it_time: %f (%f) ; eta: %s ; loss: %f (%f)' % (epoch, iteration, num_batches, time_end - time_start, accumulated_iteration_time(), eta, loss.item(), accumulated_loss()))
          metrics = evaluate_predictions(predictions, depths)
          self.write_metrics(metrics, wandb_step = wandb_step, train = True)

          test_images, test_depths, test_preds, test_loss, test_metrics = evaluate(model, self.dataloaders.get_val_dataloader, batch_size = config['test_batch_size']) ; model.train() # evaluate(in model.eval()) and back to train
          self.compare_predictions(test_images, test_depths, test_preds, wandb_step)
          wandb.log({'Validation loss on random batch':test_loss.item()}, step = wandb_step)
          self.write_metrics(test_metrics, wandb_step = wandb_step, train = False)

          if metrics['rmse'] < best_rmse: 
            wandb.run.summary["best_train_rmse"] = metrics['rmse']
            best_rmse = metrics['rmse']
            is_best = True

          save_checkpoint({'iteration': wandb_step, 
                          'state_dict': model.state_dict(), 
                          'optim_dict': optimizer.state_dict()},
                          is_best = is_best,
                          checkpoint_dir = 'experiments/', train = True)

          if test_metrics['rmse'] < best_test_rmse:
            wandb.run.summary["best_test_rmse"] = test_metrics['rmse'] 
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
      wandb.log({'Average Training loss across iters': accumulated_loss().item()}, step = wandb_step) 
      lr_scheduler.step() 
      torch.cuda.empty_cache()
      
      save_epoch({'state_dict': model.state_dict(), 
                  'optim_dict': optimizer.state_dict()}, epoch_index = epoch)
      
     


  def write_metrics(self, metrics, wandb_step, train = True):
    if train:
      for key, value in metrics.items():
        wandb.log({'Train '+key: value}, step = wandb_step)
    else:
      for key, value in metrics.items():
        wandb.log({'Validation '+key: value}, step = wandb_step) 


  def compare_predictions(self, images, depths, predictions, wandb_step):
    image_plots = plot_batch_images(images)
    depth_plots = plot_batch_depths(depths)
    pred_plots = plot_batch_depths(predictions)
    difference = plot_batch_depths(torch.abs(depths - predictions))

    wandb.log({"Sample Validation images": [wandb.Image(image_plot) for image_plot in image_plots]}, step = wandb_step)
    wandb.log({"Sample Validation depths": [wandb.Image(image_plot) for image_plot in depth_plots]}, step = wandb_step)
    wandb.log({"Sample Validation predictions": [wandb.Image(image_plot) for image_plot in pred_plots]}, step = wandb_step)
    wandb.log({"Sample Validation differences": [wandb.Image(image_plot) for image_plot in difference]}, step = wandb_step)
    
    del image_plots
    del depth_plots
    del pred_plots
    del difference



    


