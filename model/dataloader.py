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
  def __call__(self, sample, maxDepth = 1000.0):
    img, depth = sample['img'], sample['depth']
    img = self.to_torch(img)

    depth = depth.resize((320, 240)) # wxh,   TODO:modularise the shape  


    depth = self.to_torch(depth).float() * maxDepth  

    return {'img': img, 'depth': depth}

  def to_torch(self, x):
    """
    Takes a PIL Image, normalizes it to be between [0,1] and turns it into a Torch tensor C * H * W
    """  
    if not isinstance(x, Image.Image):
      raise TypeError('Not a PIL Image') 
    x_numpy = np.asarray(x, dtype = 'float32')

    x_torch = torch.from_numpy(x_numpy)

    if len(x_torch.shape) < 3:  # depth case
      x_torch = x_torch.view(x_torch.shape[0], x_torch.shape[1], 1)
      
    x_torch = x_torch.transpose(0, 1).transpose(0, 2).contiguous()  

    x_torch = x_torch.float().div(255)

    return x_torch


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

    if self.transforms:
      img, depth = self.transforms(img, depth)

    return img, depth  

  def __len__(self):   
    return self.length


class NYUDepthDatasetRaw(torch.utils.data.Dataset):
  def __init__(self, zip_file, dataset, transforms):
    self.zip_file = zip_file # the file is dynamically opened using BytesIO 
    self.dataset = dataset
    self.transforms = transforms
    self.maxDepth = 1000.0 

  def __getitem__(self, idx):
    sample = self.dataset[idx]
    img = Image.open(BytesIO(self.zip_file[sample[0]]))

    depth = Image.open(BytesIO(self.zip_file[sample[1]]))

    #TODO: see if required to normalise the depth as maxDepth/depth
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
  def __init__(self, path, train_val_ratio = 0.9):
    self.data = self.get_zip_file(path)
    self.nyu_train = []
    for row in self.data['data/nyu2_train.csv'].decode('UTF-8').split('\n'):
      if len(row) > 0:
        self.nyu_train.append(row.split(','))

    self.nyu_test = []
    for row in self.data['data/nyu2_test.csv'].decode('UTF-8').split('\n'):
      if len(row) > 0:
        self.nyu_test.append(row.split(','))

    random.shuffle(self.nyu_train)

    num_train = int(len(self.nyu_train) * train_val_ratio) 
    
    self.nyu_val = self.nyu_train[num_train:]
    self.nyu_train = self.nyu_train[0: num_train]      

  def get_train_dataloader(self, batch_size, shuffle = True):

    train_dataset = NYUDepthDatasetRaw(self.data, self.nyu_train, get_train_transforms())
    train_dataloader = torch.utils.data.DataLoader(train_dataset, 
                                                  batch_size = batch_size,
                                                  shuffle = shuffle,
                                                  num_workers = 8) 
    return train_dataloader


  def get_val_dataloader(self, batch_size, shuffle = False):
    val_dataset = NYUDepthDatasetRaw(self.data, self.nyu_val, get_test_transforms())
    val_dataloader = torch.utils.data.DataLoader(val_dataset, 
                                                  batch_size = batch_size,
                                                  shuffle = shuffle,
                                                  num_workers = 2)
    return val_dataloader                                              


  def get_zip_file(self, path):
    input_zip = ZipFile(path)
    data = {name: input_zip.read(name) for name in input_zip.namelist()}
    return data   