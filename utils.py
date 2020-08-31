import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import os 
import shutil
import torch

def plot_color(ax, color, title="Color"):

    ax.axis('off')
    ax.imshow(color)

    return ax

def plot_depth(ax, depth, title="Depth"):
    
    ax.axis('off')
    ax.imshow(depth, cmap = 'jet')

    return ax


def plot_sample_tensor(img, depth):
  """
  Accepts Torch tensors and plots them 
  """
  img = img.cpu().numpy().transpose(1,2,0) * 255
  depth = depth.cpu().numpy().transpose(1,2,0) * 255

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
  predicted_depth = predicted_depth.cpu().numpy().transpose(1,2,0) * 255
  true_depth = true_depth.cpu().numpy().transpose(1,2,0) * 255

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
    depth = depth.cpu().numpy().transpose(1,2,0) * 255
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


def save_checkpoint(state, checkpoint_dir):
  filename = 'last.pth.tar'
  if not os.path.isdir(checkpoint_dir): os.mkdir(checkpoint_dir)
  torch.save(state, os.path.join(checkpoint_dir, filename))

def load_checkpoint(checkpoint, model, optimizer=None):
    if not os.path.exists(checkpoint):
        raise Exception("File doesn't exist {}".format(checkpoint))
    checkpoint = torch.load(checkpoint)
    print('Restoring checkpoint from net iteration %d' % (checkpoint['iteration']))
    # print('Restoring checkpoint from end of epoch %d' % (checkpoint['epoch']))
    model.load_state_dict(checkpoint['state_dict'])
    if optimizer: optimizer.load_state_dict(checkpoint['optim_dict'])

def normalize_batch(batch):
    '''Normalize a tensor in [0,1] using imagenet mean and std'''    
    mean = batch.new_tensor([0.485, 0.456, 0.406]).view(-1, 1, 1)
    std = batch.new_tensor([0.229, 0.224, 0.225]).view(-1, 1, 1)
    return (batch - mean) / std
