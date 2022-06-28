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
import tqdm
import traceback

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

def evaluation_augmentations(original_shape:tuple, target_shape=256):
    return A.Compose([
        A.Resize(target_shape,target_shape,p=1),
    ], additional_targets={'dct':'image', 'sb':'image'})

def pytorch_tranformations():
    return A.Compose([
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=255.0), 
            ToTensorV2()
        ], additional_targets={'dct':'image', 'sb':'image'})


class Columbia_IFSTC(Dataset):
    def __init__(self, dataset_root:str, dataframe_path:str, transform=None, split=['test'],
                 image_size=256, to_tensor=True, shuffle=False, verbose=False, dct_flag=False, sb_flag=False):
        self.dataset_root = dataset_root
        self.dataframe_path = dataframe_path
        self.split = split
        self.image_size = image_size
        self.transform = transform
        self.shuffle = shuffle
        self.dct_flag = dct_flag
        self.sb_flag = sb_flag
        
        self.config_dataset()
        self.to_tensor = to_tensor
        self.verbose = verbose
        self.tensor_transform = pytorch_tranformations()
        
    def config_dataset(self):
        self.dataframe = pd.read_csv(self.dataframe_path)
        if self.split is not None:
            self.dataframe = self.dataframe.loc[self.dataframe.split==self.split]
        self.dataframe['image_path'] = self.dataset_root+self.dataframe.image
        self.images = self.dataframe['image_path'].tolist()
        
        self.dataframe['mask_path'] = self.dataset_root+self.dataframe['mask']
        self.masks = self.dataframe['mask_path'].tolist()
        
        if self.dct_flag:
            self.dataframe['dct_path'] = self.dataset_root+self.dataframe['dct']
            self.dcts = self.dataframe['dct_path'].tolist()
            
        if self.sb_flag:
            self.dataframe['sb_path'] = self.dataset_root+self.dataframe['sb']
            self.sbs = self.dataframe['sb_path'].tolist()
        
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
        new_mask[indices]=max_value
        return new_mask
    
    def read_images(self, idx):
        image_path = self.images[idx]
        mask_path = self.masks[idx]
        if self.dct_flag:
            dct_path = self.dcts[idx]
        if self.sb_flag:
            sb_path = self.sbs[idx]
        image = np.array(Image.open(image_path).convert('RGB'))
        mask = np.array(Image.open(mask_path).convert('L')).astype(np.float32)
        if self.dct_flag:
            dct = np.array(Image.open(dct_path).convert('RGB'))
        if self.sb_flag:
            sb = np.array(Image.open(sb_path).convert('RGB'))
        data={}
        data['image'] = image
        original_h, original_w = image.shape[:2]
        data['mask'] = mask
        if self.dct_flag:
            data['dct'] = dct
        if self.sb_flag:
            data['sb'] = sb
        if self.transform is not None:
            data = self.transform((original_h, original_w), self.image_size)(**data)
        data['mask'] = self.binarize_mask(data['mask'])
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