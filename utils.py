import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

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
  depth = depth.cpu().detach().numpy().transpose(1,2,0) / 1000 * 255 #detach is used as requires_grad error appears ----> call this function within torch.no_grad() scope

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

  def update(self, value, n_items):
    self.sum += value * n_items
    self.count += n_items

  def __call__(self):
    return self.sum/self.count  
