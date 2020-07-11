"""
TODO: 
change metrics to test metrics
add image comparison in tensorboard
"""
import time
import datetime


import numpy as np
import torch 
import torch.nn as nn
import torch.nn.functional as F
import torchvision.utils as vutils # contains useful functions like make_grid
from torch.utils.tensorboard import SummaryWriter


from model.net import DenseDepth, evaluate_predictions, combined_loss 
from model.dataloader import DataLoaders
from utils import *
from evaluate import evaluate

# shift these to config files or inside the class later
DATA_PATH = 'data/raw/nyu_data.zip'
NUM_EPOCHS = 1
LEARNING_RATE = 1e-4


class Trainer():
  def __init__(self):
    self.dataloaders = DataLoaders(DATA_PATH)  

  def train_and_evaluate(self, batch_size):
    """
    TODO: log other values/images
    """
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    train_dataloader = self.dataloaders.get_train_dataloader(batch_size = batch_size) # provide val batch size also
    num_batches = len(train_dataloader)

    model = DenseDepth()
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), LEARNING_RATE)

    # lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size = 200, gamma = 0.1)
    writer = SummaryWriter(comment = 'densenet121-bs-{}-lr-{}-epochs-{}'.format(batch_size, LEARNING_RATE, NUM_EPOCHS), flush_secs = 30)

    best_rmse = 9e20
    is_best = False

    for epoch in range(NUM_EPOCHS):
      model.train()
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


        net_iteration_number = epoch * num_batches + iteration

        if iteration % 10 == 0: 
          writer.add_scalar('Training loss wrt iterations',loss, net_iteration_number)

        if iteration % 50 == 0:

          print('Epoch: %d [%d / %d] ; it_time: %f (%f) ; eta: %s ; loss: %f (%f)' % (epoch, iteration, num_batches, time_end - time_start, accumulated_iteration_time(), eta, loss, accumulated_loss()))
          metrics = evaluate_predictions(predictions, depths)
          self.write_metrics(writer, metrics, net_iteration_number, train = True)

          test_images, test_depths, test_preds, test_loss, test_metrics = evaluate(model, self.dataloaders.get_val_dataloader, batch_size = 2) # evaluate on a random batch in the test dataloader
          self.compare_predictions(writer, test_images, test_depths, test_preds)
          writer.add_scalar('Val loss on one random batch wrt iterations',test_loss, net_iteration_number)
          self.write_metrics(writer, test_metrics, net_iteration_number, train = False)

          if metrics['rmse'] < best_rmse: #change this to test metrics later
            best_rmse = metrics['rmse']
            is_best = True

          save_checkpoint({'iteration':net_iteration_number, 
                          'state_dict': model.state_dict(), 
                          'optim_dict': optimizer.state_dict()},
                          is_best = is_best,
                          checkpoint_dir = 'experiments/train')

      epoch_end_time = time.time()
      print('****Epoch: %d ; epoch_time: %f ; av_loss: %f ****' % (epoch, epoch_end_time - epoch_start_time, accumulated_loss()))
      writer.add_scalar('Average Training loss wrt epochs', accumulated_loss(), epoch)  
      
     


  def write_metrics(self, writer, metrics, iteration_num, train = True):
    if train:
      for key, value in metrics.items():
        writer.add_scalar('Train_'+key, value, iteration_num)
    else:
      for key, value in metrics.items():
        writer.add_scalar('Val_'+key, value, iteration_num) 


  def compare_predictions(self, writer, images, depths, predictions)         
    # Plots the image on Tensorboard along with its true depth and prediction depths, and the L1 loss image

    vis_depths = depths/1000 * 255
    vis_preds = predictions/1000 * 255
    writer.add_image('Image', vutils.make_grid(images, nrow = 2))
    writer.add_image('True depth', vutils.make_grid(vis_depths, nrow = 2))
    writer.add_image('Predicted depth', vutils.make_grid(vis_preds, nrow = 2))
    writer.add_image('L1 loss', vutils.make_grid(torch.abs(vis_depths - vis_depths), nrow = 2))


