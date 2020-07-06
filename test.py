import matplotlib.pyplot as plt
import numpy as np
from model.dataloader import *
from PIL import Image
import skimage.io as io
from zipfile import ZipFile
import torchvision.transforms as T

def get_zip_file(path):
  input_zip = ZipFile(path)
  data = {name: input_zip.read(name) for name in input_zip.namelist()}
  return data



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

def get_train_transforms():
  return T.Compose([ToTensor()])


def main(data):
  # path = 'data/labelled/nyu_depth_v2_labeled.mat'

  # train_dset = dataloader.NYUDepthDatasetLabelled(path, 'train', 0.8, None)
  # img, depth = train_dset[0]

  # fig = plt.figure("Labeled Dataset Sample", figsize=(12, 5))
  # ax = fig.add_subplot(1, 2, 1)
  # plot_color(ax, img)
  # plt.show()

  # io.imshow(np.asarray(depth))
  # io.show()

  train_val_ratio = 0.8

  nyu_train = []
  for row in data['data/nyu2_train.csv'].decode('UTF-8').split('\n'):
    if len(row) > 0:
      nyu_train.append(row.split(',')) # stores the image and its depth

  nyu_test = []
  for row in data['data/nyu2_test.csv'].decode('UTF-8').split('\n'):
    if len(row) > 0:
      nyu_test.append(row.split(',')) # stores the image and its depth    

  num_train = int(len(nyu_train) * train_val_ratio) # the number of training examples to use

  nyu_val = nyu_train[num_train :]
  nyu_train = nyu_train[0: num_train]

  dset = NYUDepthDatasetRaw(data, nyu_train, get_train_transforms())
  sample = dset[0]
  img, depth = sample['img'], sample['depth']


  # Below applied for PIL Image
  
  # img = np.asarray(img, dtype = 'float32') / 255.0
  # depth = np.asarray(depth, dtype = 'float32') / 255.0

  # img = img * 255
  # depth = depth * 255


  # Below for tensor

  # img = img.numpy().transpose(1,2,0) * 255
  # depth = depth.numpy().transpose(1,2,0) * 255 / 1000

  # img = Image.fromarray(img.astype(np.uint8), mode = 'RGB')
  # depth = Image.fromarray(depth.astype(np.uint8)[:,:,0], mode='L')



  # Plot

  # fig = plt.figure("Raw Dataset Sample", figsize=(12, 5))
  # ax = fig.add_subplot(1, 2, 1)
  # plot_color(ax, img)
  # plt.show()

  # io.imshow(np.asarray(depth))
  # io.show()


