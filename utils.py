import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import os 
import shutil
import torch


def plot_color(ax, color, title="Color"):

    ax.axis('off')
    ax.set_title(title)
    ax.imshow(color)

def plot_depth(ax, depth, title="Depth"):
    
    ax.axis('off')
    ax.set_title(title)
    ax.imshow(depth, cmap = 'jet')

def plot_sample_tensor(img, depth):
  """
  Accepts Torch tensors and plots them 
  """
  img = img.cpu().numpy().transpose(1,2,0) * 255
  # print('Range of depth: ', depth.cpu().numpy().min(), depth.cpu().numpy().max())
  depth = (depth.cpu().numpy().transpose(1,2,0) / 1000) * 255

  img = Image.fromarray(img.astype(np.uint8), mode = 'RGB')
  depth = Image.fromarray(depth.astype(np.uint8)[:,:,0], mode='L')

  fig = plt.figure("Sample", figsize=(12, 5))

  ax = fig.add_subplot(1, 2, 1)
  plot_color(ax, img)

  ax = fig.add_subplot(1, 2, 2)
  plot_depth(ax, depth)

def plot_sample_image(img, depth):

  fig = plt.figure("Sample", figsize=(12, 5))

  ax = fig.add_subplot(1, 2, 1)
  plot_color(ax, img)

  ax = fig.add_subplot(1, 2, 2)
  plot_depth(ax, depth)


class RunningAverage():
  def __init__(self):
    self.count = 0
    self.sum = 0

  def update(self, value, n_items = 1):
    self.sum += value * n_items
    self.count += n_items

  def __call__(self):
    return self.sum/self.count  


def save_checkpoint(state, is_best, checkpoint_dir):
    """Saves model and training parameters at checkpoint + 'last.pth.tar'. If is_best==True, also saves
    checkpoint + 'best.pth.tar'
    Args:
        state: (dict) contains model's state_dict, may contain other keys such as epoch, optimizer state_dict
        is_best: (bool) True if it is the best model seen till now
        checkpoint: (string) folder where parameters are to be saved
    """
    filepath = os.path.join(checkpoint_dir, 'last.pth.tar')
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(checkpoint_dir, 'best.pth.tar'))

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