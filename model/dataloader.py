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


def get_test_data(path):
  """
  Returns all the torch tensors loading the image([0,1]), depth([-1,1]), eigen_crop(4 co-ordinates) of test data ZIP FILE
  """
  input_zip = ZipFile(path)
  data = {name: input_zip.read(name) for name in input_zip.namelist()}

  rgb = np.load(BytesIO(data['eigen_test_rgb.npy']))
  depth = np.load(BytesIO(data['eigen_test_depth.npy']))
  crop = np.load(BytesIO(data['eigen_test_crop.npy'])) 
  depth = np.clip(depth, 1.0, 10.0) / 10 * 255 

  toTensorFunc = ToTensor()
  samples = []
  for i in range(rgb.shape[0]):
    img = Image.fromarray(rgb[i].astype(np.uint8), mode = 'RGB')
    img_depth = Image.fromarray(depth[i].astype(np.uint8)[:,:], mode='L')
    sample = {'img':img, 'depth':img_depth}
    samples.append(toTensorFunc(sample = sample))
    
  return samples, torch.from_numpy(crop)    


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
  def __init__(self, dataset, transforms, resized):
    self.dataset = dataset
    self.transforms = transforms
    self.resized = resized

  def __getitem__(self, idx):
    sample = self.dataset[idx]
    img = Image.open(sample[0])

    depth = Image.open(sample[1])
    if self.resized:
      depth = depth.resize((320, 240)) # wxh
       
    datum = {'img': img, 'depth': depth}
    if self.transforms:
      datum = self.transforms(datum) # a composed set of transforms should be passed

    return datum  

  def __len__(self):   
    return len(self.dataset)


def get_train_transforms():
  return T.Compose([RandomHorizontalFlip(), RandomChannelSwap(), ToTensor()])

def get_test_transforms():
  return T.Compose([ToTensor()])


class DataLoaders:
  def __init__(self, data_dir, resized = True):    
    self.nyu_train = []
    for row in csv.reader(open(os.path.join(data_dir, 'nyu_train.csv')), delimiter=','):
      if len(row) > 0:
        self.nyu_train.append(row.split(','))
    self.nyu_val = []
    for row in csv.reader(open(os.path.join(data_dir, 'nyu_val.csv')), delimiter=','):
      if len(row) > 0:
        self.nyu_val.append(row.split(','))
    self.resized = resized  

  def get_train_dataloader(self, batch_size, shuffle = True):

    train_dataset = NYUDepthDatasetRaw(self.nyu_train, get_train_transforms(), self.resized)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, 
                                                  batch_size = batch_size,
                                                  shuffle = shuffle,
                                                  num_workers = 4) 
    return train_dataloader

  def get_val_dataloader(self, batch_size, shuffle = True):

    val_dataset = NYUDepthDatasetRaw(self.nyu_val, get_test_transforms(), self.resized)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, 
                                                  batch_size = batch_size,
                                                  shuffle = shuffle,
                                                  num_workers = 4) 
    return val_dataloader