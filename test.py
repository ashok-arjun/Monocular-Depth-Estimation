import matplotlib.pyplot as plt
import numpy as np
import model.dataloader as dataloader
from PIL import Image
import skimage.io as io

def plot_color(ax, color, title="Color"):
    """Displays a color image from the NYU dataset."""

    ax.axis('off')
    ax.set_title(title)
    ax.imshow(color)

def plot_depth(ax, depth, title="Depth"):
    """Displays a depth map from the NYU dataset."""

    
    ax.axis('off')
    ax.set_title(title)
    ax.imshow(depth)

def main():
  path = 'data/labelled/nyu_depth_v2_labeled.mat'

  train_dset = dataloader.NYUDepthDatasetLabelled(path, 'train', 0.8, None)
  img, depth = train_dset[0]

  fig = plt.figure("Labeled Dataset Sample", figsize=(12, 5))
  ax = fig.add_subplot(1, 2, 1)
  plot_color(ax, img)
  plt.show()

  io.imshow(np.asarray(depth))
  io.show()