import os
import pathlib

import torch
import torchvision.transforms as T
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

import cs3600.download

torch.manual_seed(42)


class MyDataSet():
    def __init__(self):
        DATA_DIR = pathlib.Path.home().joinpath('.pytorch-datasets')
        DATA_URL = 'http://vis-www.cs.umass.edu/lfw/lfw-bush.zip'

        _, dataset_dir = cs3600.download.download_data(out_path=DATA_DIR, url=DATA_URL, extract=True, force=False)

        im_size = 64
        tf = T.Compose([
            # Resize to constant spatial dimensions
            T.Resize((im_size, im_size)),
            # PIL.Image -> torch.Tensor
            T.ToTensor(),
            # Dynamic range [0,1] -> [-1, 1]
            T.Normalize(mean=(.5, .5, .5), std=(.5, .5, .5)),
        ])
        
        self.ds_gwb = ImageFolder(os.path.dirname(dataset_dir), tf)
           
    @property
    def image_dim(self): 
        return self.ds_gwb[0][0].shape