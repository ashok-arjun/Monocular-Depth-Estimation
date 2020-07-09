import numpy as np
import torch 
import torch.nn as nn
import torch.nn.functional as F
import torchvision.utils as vutils # contains useful functions like make_grid
from torch.utils.tensorboard import SummaryWriter


from model.net import DenseDepth, evaluate_predictions, mean_l1_loss, gradient_loss, ssim # model, metrics and loss functions
from model.dataloader import DataLoaders
from utils import *

# shift these to config files or inside the class later
DATA_PATH = 'data/raw/nyu_data.zip'
BATCH_SIZE = 2
NUM_EPOCHS = 1
LEARNING_RATE = 1e-4


class Trainer():
  def __init__(self):
    print('Decoding the zip file...')
    self.dataloaders = DataLoaders(DATA_PATH)  
    print('Done decoding the zip file...')

  def train_and_evaluate(self):
    """
    Trains the DenseDepth model for NUM_EPOCHS,
    Uses a combination of all the losses,
    Uses the evaluate_predictions to get metrics,
    Validation data is  evaluated on loss and metrics every epoch
    Training data is evaluated on loss every iteration, on metrics every epoch
    After every iteration/epoch, loss is logged to the Tensorboard.
    CHECKPOINT THE MODEL AT EVERY EPOCH/ITERATION

    TODO: log other values/images
    """
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    train_dataloader, val_dataloader, test_dataloader = self.dataloaders.get_dataloaders(train_batch_size = BATCH_SIZE) # provide val batch size also
    num_batches = len(train_dataloader)

    model = DenseDepth()
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), LEARNING_RATE)

    # lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size = 200, gamma = 0.1)

    writer = SummaryWriter(log_dir = 'experiments/densenet121-bs-{}-lr-{}-epochs-{}'.format(BATCH_SIZE, LEARNING_RATE, NUM_EPOCHS), flush_secs = 30)

    print('Training the model...')
    for epoch in range(NUM_EPOCHS):
      print('Epoch:-', epoch)
      model.train()
      accumulated_loss = RunningAverage()
      # optionally, average iteration time also

      for iteration, batch in enumerate(train_dataloader):
        

        optimizer.zero_grad()
        images, depths = batch['img'], batch['depth']
        images = torch.autograd.Variable(images.to(device))
        depths = torch.autograd.Variable(depths.to(device))

        # depths = 1000.0/depths

        predictions = model(images)

        loss = mean_l1_loss(predictions, depths) + gradient_loss(predictions, depths) # add SSIM loss here, after observing the current loss and outputs
        accumulated_loss.update(loss, images.shape[0])

        loss.backward()
        optimizer.step()

        net_iteration_number = epoch * num_batches + iteration
        if iteration % 10 == 0:
          writer.add_scalar('Training loss wrt iterations',loss, net_iteration_number)

      print('Average loss:- ', accumulated_loss())
      writer.add_scalar('Training loss wrt epochs', accumulated_loss(), epoch)  
      # Validation loss - evaluate
      # Validation metrics - evaluate
     


    

