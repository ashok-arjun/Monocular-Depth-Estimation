"""
The .zip file is extracted online, and the train and test dataframes are split
"""

import numpy as np
import os
import h5py
import torch
import torchvision
import torchvision.transforms as T
from PIL import Image

# Raw dataset class to be created

def train_transform(image, depth):
  img_transforms = T.Compose([
    T.Resize((640, 480)),
    T.ToTensor()
  ])
  depth_transforms = T.Compose([
    T.Resize((320, 240)),
    T.ToTensor()
  ])
  
  return img_transforms(image), depth_transforms(depth)

class NYUDepthDatasetLabelled(torch.utils.data.Dataset):
  def __init__(self, path, split, split_ratio, transforms):
    self.mat = h5py.File(path, 'r')
    num_train = int(self.mat['images'].shape[0] * split_ratio)
    if split == 'train':
      self.start_offset = 0
      self.length = num_train 
    else:
      self.start_offset = num_train
      self.length = self.mat['images'].shape[0] - num_train 
    self.transforms = transforms

  def __getitem__(self, idx):
    img_mat = self.mat['images'][self.start_offset + idx]
    img = np.empty([480, 640, 3])
    img[:,:,0] = img_mat[0,:,:].T
    img[:,:,1] = img_mat[1,:,:].T
    img[:,:,2] = img_mat[2,:,:].T
    img = img.astype('uint8')
    img = Image.fromarray(img)

    depth_mat = self.mat['depths'][idx]
    depth = np.empty([480, 640])
    depth = depth_mat.T
    depth = depth.astype('uint8')
    depth = Image.fromarray(depth)

    # to display depth, divide by 4
    if self.transforms:
      img, depth = self.transforms(img, depth)

    return img, depth  

  def __len__(self):   
    return self.length