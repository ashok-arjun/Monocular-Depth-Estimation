import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import os 
import shutil
import torch
import wandb


def plot_color(ax, color, title="Color"):

    ax.axis('off')
    # ax.title(title)
    ax.imshow(color)

    return ax

def plot_depth(ax, depth, title="Depth"):
    
    ax.axis('off')
    # ax.title(title)
    ax.imshow(depth, cmap = 'jet')

    return ax


def plot_sample_tensor(img, depth):
  """
  Accepts Torch tensors and plots them 
  """
  img = img.cpu().numpy().transpose(1,2,0) * 255
  # print('Range of depth: ', depth.cpu().numpy().min(), depth.cpu().numpy().max())
  depth = (depth.cpu().numpy().transpose(1,2,0) / 1000) * 255

  img = Image.fromarray(img.astype(np.uint8), mode = 'RGB')
  depth = Image.fromarray(depth.astype(np.uint8)[:,:,0], mode='L')

  fig = plt.figure("Example", figsize=(12, 5))

  ax = fig.add_subplot(1, 2, 1)
  plot_color(ax, img)

  ax = fig.add_subplot(1, 2, 2)
  plot_depth(ax, depth)

  return ax

def plot_predicted_deviation(predicted_depth, true_depth):
  """
  Accepts Torch tensors and plots them 
  """
  predicted_depth = (predicted_depth.cpu().numpy().transpose(1,2,0) / 1000) * 255
  true_depth = (true_depth.cpu().numpy().transpose(1,2,0) / 1000) * 255

  diff = predicted_depth - true_depth

  predicted_depth = Image.fromarray(predicted_depth.astype(np.uint8)[:,:,0], mode='L')
  diff = Image.fromarray(diff.astype(np.uint8)[:,:,0], mode='L')

  fig = plt.figure("Example", figsize=(12, 5))

  ax = fig.add_subplot(1, 2, 1)
  plot_depth(ax, predicted_depth, title = 'Predicted depth')

  ax = fig.add_subplot(1, 2, 2)
  plot_depth(ax, diff, title='Difference')

  return ax  

def plot_sample_image(img, depth):

  fig = plt.figure("Sample", figsize=(12, 5))

  ax = fig.add_subplot(1, 2, 1)
  plot_color(ax, img)

  ax = fig.add_subplot(1, 2, 2)
  plot_depth(ax, depth)


def plot_batch_images(images):
  plots = []

  for img in images:
    img = img.cpu().numpy().transpose(1,2,0) * 255
    img = Image.fromarray(img.astype(np.uint8), mode = 'RGB')
    plots.append(img)

  return plots  

def plot_batch_depths(depths):
  plots = []
  for depth in depths:
    depth = (depth.cpu().numpy().transpose(1,2,0) / 1000) * 255
    depth = Image.fromarray(depth.astype(np.uint8)[:,:,0], mode='L')
    plots.append(depth)  

  return plots 


class RunningAverage():
  def __init__(self):
    self.count = 0
    self.sum = 0

  def update(self, value, n_items = 1):
    self.sum += value * n_items
    self.count += n_items

  def __call__(self):
    return self.sum/self.count  


def save_checkpoint(state, is_best, checkpoint_dir, train = True):
    """Saves model and training parameters at checkpoint + 'last.pth.tar'. If is_best==True, also saves
    checkpoint + 'best.pth.tar'
    Args:
        state: (dict) contains model's state_dict, may contain other keys such as epoch, optimizer state_dict
        is_best: (bool) True if it is the best model seen till now
        checkpoint: (string) folder where parameters are to be saved
    """
    prefix = 'train' if train else 'test'
    torch.save(state, os.path.join(checkpoint_dir, prefix + '_last.pth.tar'))
    torch.save(state, os.path.join(wandb.run.dir, prefix + "_last.pth.tar"))
#     wandb.save(prefix + '_last.pth.tar')
    if is_best:
        torch.save(state, os.path.join(checkpoint_dir, prefix + '_best.pth.tar'))
        torch.save(state, os.path.join(wandb.run.dir, prefix + "_best.pth.tar"))
#         wandb.save(prefix + '_best.pth.tar')

def load_checkpoint(checkpoint, model, optimizer=None):
    """Loads model parameters (state_dict) from file_path. If optimizer is provided, loads state_dict of
    optimizer assuming it is present in checkpoint.
    Args:
        checkpoint: (string) filename which needs to be loaded
        model: (torch.nn.Module) model for which the parameters are loaded
        optimizer: (torch.optim) optional: resume optimizer from checkpoint
    """
    if not os.path.exists(checkpoint):
        raise("File doesn't exist {}".format(checkpoint))
    checkpoint = torch.load(checkpoint)
    model.load_state_dict(checkpoint['state_dict'])

    if optimizer:
        optimizer.load_state_dict(checkpoint['optim_dict'])

    return checkpoint

def save_epoch(state, epoch_index):
    prefix = str(epoch_index)
    print('Trying to save epoch', prefix)
    torch.save(state, os.path.join(wandb.run.dir, 'epoch_' + prefix + ".pth.tar"))
    print('Epoch saved to cloud: ', prefix)
    wandb.save('epoch_' + prefix + ".pth.tar")
