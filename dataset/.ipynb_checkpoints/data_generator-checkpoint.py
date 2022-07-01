import pandas as pd
import os
import numpy as np
import PIL
from PIL import Image
import torch
from torch import nn
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2
import math 
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.model_selection import train_test_split
from skimage import io, transform
from skimage.util import view_as_windows
from skimage import img_as_int, img_as_ubyte
import skimage
from scipy import signal
from torchvision import transforms, utils, datasets
import pyfftw
import tqdm



def train_augmentations(original_shape:tuple, target_shape=256):
    return A.Compose([
        A.Resize(original_shape[0],original_shape[1],p=1),
        A.SmallestMaxSize(max_size=target_shape),
        A.RandomCrop(target_shape, target_shape, always_apply=False, p=1),
        A.HorizontalFlip(p=0.5)
    ], additional_targets={'dct':'image', 'sb':'image'})

def val_augmentations(original_shape:tuple, target_shape=256):
    return A.Compose([
        A.Resize(original_shape[0],original_shape[1],p=1),
        A.SmallestMaxSize(max_size=target_shape),
        A.CenterCrop(target_shape, target_shape, always_apply=False, p=1),
    ], additional_targets={'dct':'image', 'sb':'image'})

def eval_augmentations(original_shape:tuple, target_shape=256):
    return A.Compose([
        A.Resize(original_shape[0],original_shape[1],p=1),
    ], additional_targets={'dct':'image', 'sb':'image'})

def pytorch_tranformations():
    return A.Compose([
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=255.0), 
            ToTensorV2()
        ], additional_targets={'dct':'image', 'sb':'image'})

import traceback
class Data_Generator(Dataset):  
    
    def __init__(self,dataset_root:str, dataframe_path:str, transform=None, split=['train'], image_size=256, 
                 to_tensor=True, shuffle=False, verbose=False, dct=False, sb=False, inverse=False):
        self.dataset_root = dataset_root
        self.dataframe_path = dataframe_path
        self.split = split
        self.image_size = image_size
        self.transform = transform
        self.shuffle = shuffle
        self.config_dataset()
        self.to_tensor = to_tensor
        self.verbose = verbose
        self.tensor_transform = pytorch_tranformations()
        self.radon = radon
        self.evaluation = evaluation
        self.flag_dct = dct
        self.flag_sb = sb
        
    def config_dataset(self):
        self.dataframe = pd.read_csv(self.dataframe_path)
        self.dataframe = self.dataframe.loc[self.dataframe.split.isin(self.split)]
        if 'val' in self.split:
            self.dataframe = self.dataframe[:100]
        self.dataframe['image_path'] = self.dataset_root+self.dataframe.image
        self.images = self.dataframe['image_path'].tolist()
        
        self.dataframe['mask_path'] = self.dataset_root + self.dataframe['mask']
        self.masks = self.dataframe['mask_path'].tolist()
        
        if self.flag_dct:
            self.dataframe['dct_path'] = self.dataset_root+self.dataframe['dct']
            self.dcts = self.dataframe['dct_path'].tolist()
        
        if self.flag_sb:
            self.dataframe['sb_path'] = self.dataset_root+self.dataframe['sb']
            self.sbs = self.dataframe['sb_path'].tolist()
        
        if self.shuffle:
            from skearn.utils import shuffle
            self.paths = shuffle(np.array(self.paths))
            
    def __len__(self):
        return len(self.images)
    
    def binarize_mask(self, mask, upper=True):
        mask = cv2.normalize(mask, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        mask = np.ceil(mask).astype(np.int32)
        return mask
    
    def inverse_mask(self, mask):
        max_value = mask.max()
        new_mask = np.zeros_like(mask)
        indices = np.where(mask==0)
        new_mask[indices] = max_value
        return new_mask
    
    def read_images(self, idx:int) -> dict:
        image_path = self.images[idx]
        mask_path = self.masks[idx]
        if self.flag_dct:
            dct_path = self.dcts[idx]
        if self.flag_sb:
            sb_path = self.sbs[idx]
        image = np.array(Image.open(image_path).convert('RGB'))
        mask = np.array(Image.open(mask_path).convert('L')).astype(np.float32)
        if self.flag_dct:
            dct = np.array(Image.open(dct_path).convert('RGB'))
        if self.flag_sb:
            sb = np.array(Image.open(sb_path).convert('RGB'))
        data={}
        data['image'] = image
        original_h, original_w = image.shape[:2]
        data['mask'] = mask
        if self.flag_dct:
            data['dct'] = dct
        if self.flag_sb:
            data['sb'] = sb
        if self.transform is not None:
            data = self.transform((original_h, original_w), self.image_size)(**data)
        data['mask'] = self.binarize_mask(data['mask'])
        if inverse:
            data['mask'] = self.inverse_mask(data['mask'])
        return data
    
    def __getitem__(self, idx):
        while 1:
            try:
                data = self.read_images(idx)
                if self.to_tensor:
                    data = self.tensor_transform(**data)
                    image = data.pop('image')
                    mask = data.pop('mask')
                    
                    if(self.flag_dct and self.flag_sb):
                        dct = data.pop('dct')
                        sb = data.pop('sb')
                        return image, mask, dct, sb                      
                    if self.flag_dct:
                        dct = data.pop('dct')
                        return image, mask, dct
                    if self.flag_sb:
                        sb = data.pop('sb')
                        return image, mask, sb
                else:
                    return data
            except Exception as e:
                if self.verbose:
                    traceback.print_exc()
                idx = np.random.randint(0, len(self))





