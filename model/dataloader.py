import numpy as np
import os
import h5py
import torch
import torchvision
import torchvision.transforms as T
from PIL import Image
from zipfile import ZipFile
from io import BytesIO
import random
from itertools import permutations
import csv


def get_test_dataloader(data_dir, batch_size):
  dataset = NYUDepthTestDataset(data_dir, get_test_tranforms())
  dataloader = torch.utils.data.DataLoader(dataset, 
                                          batch_size = batch_size,
                                          shuffle = False,
                                          num_workers = 0) 
  return dataloader


class RandomHorizontalFlip(object):
  def __init__(self, prob = 0.5):
    self.prob = prob
  def __call__(self, sample):
    img, depth = sample['img'], sample['depth']

    assert isinstance(img, Image.Image), 'img is not a PIL Image'
    assert isinstance(depth, Image.Image), 'depth is not a PIL Image'  

    if random.random() < self.prob:
      img = img.transpose(Image.FLIP_LEFT_RIGHT)
      depth = depth.transpose(Image.FLIP_LEFT_RIGHT)

    return {'img': img, 'depth': depth}  


class RandomChannelSwap(object):
  def __init__(self, prob = 0.5):
    self.prob = prob
    self.channel_perms = list(permutations(range(3)))
  def __call__(self, sample):
    img, depth = sample['img'], sample['depth']

    assert isinstance(img, Image.Image), 'img is not a PIL Image'
    assert isinstance(depth, Image.Image), 'depth is not a PIL Image'  

    if random.random() < self.prob:
      img = np.asarray(img)
      img = Image.fromarray(img[..., list(self.channel_perms[random.randint(0, len(self.channel_perms) - 1)]) ])

    return {'img': img, 'depth': depth}      


class ToTensor(object):
  '''Receives input as numpy arrays/PIL images and depths in range 0,255 and converts them to 0,1'''
  def __call__(self, sample):
    img, depth = sample['img'], sample['depth']
    img = self.to_torch(img)

    depth = self.to_torch(depth).float() 

    return {'img': img, 'depth': depth}

  def to_torch(self, x):
    """
    Takes a PIL Image/numpy array, normalizes it to be between [0,1] and turns it into a Torch tensor C * H * W
    """  
    x_numpy = np.asarray(x, dtype = 'float32')

    x_torch = torch.from_numpy(x_numpy)

    if len(x_torch.shape) < 3:  # depth case
      x_torch = x_torch.view(x_torch.shape[0], x_torch.shape[1], 1)
      
    x_torch = x_torch.transpose(0, 1).transpose(0, 2).contiguous()  

    x_torch = x_torch.float().div(255)

    return x_torch

class NYUDepthDatasetRaw(torch.utils.data.Dataset):
  def __init__(self, data_dir, dataset, transforms, resized):
    self.data_dir = data_dir
    self.dataset = dataset
    self.transforms = transforms
    self.resized = resized

  def __getitem__(self, idx):
    sample = self.dataset[idx]
    img = Image.open(os.path.join(self.data_dir, sample[0]))

    depth = Image.open(os.path.join(self.data_dir, sample[1]))
    if self.resized:
      depth = depth.resize((320, 240)) # wxh
       
    datum = {'img': img, 'depth': depth}
    if self.transforms:
      datum = self.transforms(datum) # a composed set of transforms should be passed

    return datum  

  def __len__(self):   
    return len(self.dataset)

class NYUDepthTestDataset(torch.utils.data.Dataset):
  def __init__(self, data_dir, transforms):
    self.data_dir = data_dir
    self.transforms = transforms
    
    self.rgb = np.load(os.path.join(data_dir, 'eigen_test_rgb.npy'))
    self.depth = np.clip(np.load(os.path.join(data_dir, 'eigen_test_depth.npy')), 1.0, 10.0) / 10 * 255 
    self.crop = np.load(os.path.join(data_dir, 'eigen_test_crop.npy')) 
        

  def __getitem__(self, i):
    img = Image.fromarray(self.rgb[i].astype(np.uint8), mode = 'RGB')
    img_depth = Image.fromarray(self.depth[i].astype(np.uint8)[:,:], mode='L')
    sample = {'img':img, 'depth':img_depth}
    sample = self.transforms(sample)
    return sample, self.crop
    
  def __len__(self):   
    return self.rgb.shape[0]  

def get_train_transforms():
  return T.Compose([RandomHorizontalFlip(), RandomChannelSwap(), ToTensor()])

def get_test_transforms():
  return T.Compose([ToTensor()])


class DataLoaders:
  def __init__(self, data_dir, resized = True):    
    self.nyu_train = []
    for row in csv.reader(open(os.path.join(data_dir, 'data/nyu2_train.csv')), delimiter=','):
      if len(row) > 0:
        self.nyu_train.append(row)
    self.resized = resized  
    self.data_dir = data_dir

  def get_train_dataloader(self, batch_size, shuffle = True):

    train_dataset = NYUDepthDatasetRaw(self.data_dir, self.nyu_train, get_train_transforms(), self.resized)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, 
                                                  batch_size = batch_size,
                                                  shuffle = shuffle,
                                                  num_workers = 0) 
    return train_dataloader
