import numpy as np
import os
import h5py
import torch
import torchvision
import torchvision.transforms as T
from PIL import Image
from zipfile import ZipFile

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
    img = Image.fromarray(img, 'RGB')

    depth_mat = self.mat['depths'][self.start_offset + idx]
    depth = np.empty([480, 640])
    depth = depth_mat.T
    depth = Image.fromarray(depth, 'F')

    # to display depth, divide by 4
    if self.transforms:
      img, depth = self.transforms(img, depth)

    return img, depth  

  def __len__(self):   
    return self.length


class NYUDepthDatasetRaw(torch.utils.data.Dataset):
  def __init__(self, dataset, transforms):
    self.dataset = dataset
    self.transforms = transforms

  def __getitem__(self, idx):
    img_mat = self.mat['images'][self.start_offset + idx]
    img = np.empty([480, 640, 3])
    img[:,:,0] = img_mat[0,:,:].T
    img[:,:,1] = img_mat[1,:,:].T
    img[:,:,2] = img_mat[2,:,:].T
    img = img.astype('uint8')
    img = Image.fromarray(img, 'RGB')

    depth_mat = self.mat['depths'][self.start_offset + idx]
    depth = np.empty([480, 640])
    depth = depth_mat.T
    depth = Image.fromarray(depth, 'F')

    # to display depth, divide by 4
    if self.transforms:
      img, depth = self.transforms(img, depth)

    return img, depth  

  def __len__(self):   
    return self.length

def get_zip_file(path):
  input_zip = ZipFile(path)
  data = {name: input_zip.read(name) for name in input_zip.namelist()}
  return data

def get_raw_datasets(path, shuffle_train = True):
  zip_data = get_zip_file(path)

  nyu_train = []
  for row in data['data/nyu2_train.csv'].decode('UTF-8').split('\n'):
    if len(row) > 0:
      nyu_train.append(row.split(',')) # stores the image and its depth

  nyu_test = []
  for row in data['data/nyu2_test.csv'].decode('UTF-8').split('\n'):
    if len(row) > 0:
      nyu_test.append(row.split(',')) # stores the image and its depth    

  if shuffle_train:
    random.shuffle(nyu_train)

  # pass the dataframes to the datasets, and return dataloaders ---> decide augmentation later  



# Function for getting the data loaders(train/val/test)
# Baseline model as in Laina et. al.
# Complete dataset(the .zip file)
# Repeat the process
# Visualize the filters, literature, ...
