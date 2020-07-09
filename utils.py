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
  depth = depth.cpu().detach().numpy().transpose(1,2,0) / 1000 * 255 #detach is used as requires_grad error appears

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
    """A simple class that maintains the running average of a quantity
    
    Example:
    ```
    loss_avg = RunningAverage()
    loss_avg.update(2)
    loss_avg.update(4)
    loss_avg() = 3
    ```
    """
    def __init__(self):
        self.steps = 0
        self.total = 0
    
    def update(self, val):
        self.total += val
        self.steps += 1
    
    def __call__(self):
        return self.total/float(self.steps)  
