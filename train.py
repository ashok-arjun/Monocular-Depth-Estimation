import time
import datetime
import pytz  
import wandb

import numpy as np
import torch 
import torch.nn as nn
import torch.nn.functional as F
import torchvision.utils as vutils
from torch.utils.tensorboard import SummaryWriter


from model.net import MonocularDepthModel, MonocularDepthModelWithUpconvolution  
from model.loss import LossNetwork, combined_loss, mean_l2_loss
from model.metrics import evaluate_predictions
from model.dataloader import DataLoaders, get_test_data
from utils import *
from evaluate import infer_depth, evaluate_list

class Trainer():
  def __init__(self, data_path, test_zip_path, resized):
    self.dataloaders = DataLoaders(data_path, resized = resized)  
    self.resized = resized
#     self.test_data = get_test_data(test_zip_path) # (samples, crop)

  def train_and_evaluate(self, config, checkpoint = None):
    batch_size = config['batch_size']
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    train_dataloader = self.dataloaders.get_train_dataloader(batch_size = batch_size) 
    num_batches = len(train_dataloader)

    model = MonocularDepthModel()
    if self.resized == False:
      model = MonocularDepthModelWithUpconvolution(model)
    model = model.to(device)
    params = [param for param in model.parameters() if param.requires_grad == True]
    print('A total of %d parameters in present model' % (len(params)))
    optimizer = torch.optim.Adam(params, config['lr'])
    

    loss_model = LossNetwork().to(device)
      
    wandb_step = config['start_epoch'] * num_batches -1 

    accumulated_per_pixel_loss = RunningAverage()
    accumulated_feature_loss = RunningAverage()
    accumulated_iteration_time = RunningAverage()

    if checkpoint:
      load_checkpoint(checkpoint, model, optimizer)

    print('Training...')  

    for epoch in range(config['start_epoch'], config['epochs']):
      epoch_start_time = time.time()
      for iteration, batch in enumerate(train_dataloader):
        wandb_step += 1
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
          wandb.log({'Average per-pixel loss': accumulated_per_pixel_loss()}, step = wandb_step)
          wandb.log({'Average feature loss': accumulated_feature_loss()}, step = wandb_step)
          metrics = evaluate_predictions(predictions, depths)
          self.write_metrics(metrics, wandb_step = wandb_step, train = True)
                               
      epoch_end_time = time.time()
      print('Epoch %d complete, time taken: %s' % (epoch, str(datetime.timedelta(seconds = int(epoch_end_time - epoch_start_time)))))
      torch.cuda.empty_cache()

      save_checkpoint({
                  'iteration': wandb_step,
                  'state_dict': model.state_dict(), 	
                  'optim_dict': optimizer.state_dict()}, config['checkpoint_dir'], (epoch % config['save_to_cloud_every'] == 0))

      print('Epoch %d saved\n\n' % (epoch))

      # EVALUATE ON TEST DATA:

#       test_metrics = evaluate_list(model, self.test_data[0], self.test_data[1], config['test_batch_size'], model_upsample = True)
#       self.write_metrics(test_metrics, wandb_step, train=False)

#       random_indices = np.random.choice(len(self.test_data[0]), config['log_images_count'])
#       log_images = torch.cat([self.test_data[0][i]['img'].unsqueeze(0) for i in random_indices], dim = 0)
#       log_depths = torch.cat([self.test_data[0][i]['depth'].unsqueeze(0) for i in random_indices], dim = 0)
#       log_preds = torch.cat([infer_depth(img, model, upsample = True)[0].unsqueeze(0) for img in log_images], dim = 0)
#       self.compare_predictions(log_images, log_depths, log_preds, wandb_step)


  def write_metrics(self, metrics, wandb_step, train = True):	
    # if train:	
    #   for key, value in metrics.items():	
    #     wandb.log({'Train '+key: value}, step = wandb_step)	
    # else:	
    #   for key, value in metrics.items():	
    #     wandb.log({'Test '+key: value}, step = wandb_step) 	

    writing_metrics = ['d1_accuracy', 'rmse']

    if train:	
      for key in writing_metrics:	
        wandb.log({'Train '+key: metrics[key]}, step = wandb_step)	
    else:	
      for key in writing_metrics:	
        wandb.log({'Test '+key: metrics[key]}, step = wandb_step) 

  def compare_predictions(self, images, depths, predictions, wandb_step):	
    image_plots = plot_batch_images(images)	
    depth_plots = plot_batch_depths(depths)	
    pred_plots = plot_batch_depths(predictions)	
    difference = plot_batch_depths(torch.abs(depths.cpu() - predictions.cpu()))	

    wandb.log({"Sample Validation images": [wandb.Image(image_plot) for image_plot in image_plots]}, step = wandb_step)	
    wandb.log({"Sample Validation depths": [wandb.Image(image_plot) for image_plot in depth_plots]}, step = wandb_step)	
    wandb.log({"Sample Validation predictions": [wandb.Image(image_plot) for image_plot in pred_plots]}, step = wandb_step)	
    wandb.log({"Sample Validation differences": [wandb.Image(image_plot) for image_plot in difference]}, step = wandb_step)	

    del image_plots	
    del depth_plots	
    del pred_plots	
    del difference
