# import skimage.io as io
# import scipy.io
# import pandas as pd

# path = 'data/raw/nyu_depth_v2_labeled.mat'
# mat = scipy.io.loadmat(path)
# mat = {k:v for k, v in mat.items()}
# data = pd.DataFrame({k: pd.Series(v[0]) for k, v in mat.items()})
# data.to_csv("example.csv")


import model.dataloader as dataloader

path = 'data/labelled/nyu_depth_v2_labeled.mat'

train_dset = dataloader.NYUDepthDatasetLabelled(path, 'train', 0.8, dataloader.train_transform)
img, depth = train_dset[0]
print(img.shape, depth.shape)