import pandas as pd
import os
import numpy as np
from PIL import Image
import torch
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2
import math 
import torch.nn.functional as F
from torch import nn
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
from sklearn.metrics import precision_score, recall_score, f1_score, roc_curve, auc
import PIL


MAP_NAMES = ['DCTOutput.png', 'SBOutput.png']

def pytorch_tranformations():
    return A.Compose([
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=255.0), 
            ToTensorV2()
        ],additional_targets={i:'image' for i in MAP_NAMES})

def train_augmentations(original_shape:tuple, target_shape=256):
    return A.Compose([
        A.Resize(original_shape[0],original_shape[1],p=1),
        A.SmallestMaxSize(max_size=target_shape),
        A.RandomCrop(target_shape, target_shape, always_apply=False, p=1),
        A.HorizontalFlip(p=0.5)
    ], additional_targets={i:'image' for i in MAP_NAMES})

def val_augmentations(original_shape:tuple, target_shape=256):
    return A.Compose([
        A.Resize(original_shape[0],original_shape[1],p=1),
        A.SmallestMaxSize(max_size=target_shape),
        A.CenterCrop(target_shape, target_shape, always_apply=False, p=1),
    ], additional_targets={i:'image' for i in MAP_NAMES})

def evaluation_augmentations(original_shape:tuple, target_shape=256):
    return A.Compose([
        A.Resize(target_shape,target_shape,p=1),
    ], additional_targets={i:'image' for i in MAP_NAMES})


import traceback
class IFS_TC(Dataset):
    
    def __init__(self, dataset_root, dataframe_path, transform=None, split=['train'], image_size=256, to_tensor=False, verbose=False, dct_flag=False, sb_flag=False):
        self.dataset_root = dataset_root
        self.dataframe_path=dataframe_path
        self.split=split
        self.transform=transform
        self.image_size=image_size
        self.config_dataset()
        self.to_tensor=to_tensor
        self.verbose=verbose
        self.tensor_transform = pytorch_tranformations()
        self.evaluation = False
        self.dct_flag = dct_flag
        self.sb_flag = sb_flag
        
    def config_dataset(self):
        self.dataframe = pd.read_csv(self.dataframe_path)
        self.dataframe = self.dataframe.loc[self.dataframe.split.isin(self.split)]
        
        self.p = self.dataframe['id'].tolist()
        
    def __len__(self):
        return len(self.dataframe)
    
    def binarize_mask(self, mask, upper=True):
        mask = cv2.normalize(mask, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        mask = np.ceil(mask).astype(np.int32)
        return mask
    
    def inverse_mask(self,mask):
        max_value = mask.max()
        new_mask = np.zeros_like(mask)
        indices = np.where(mask==0)
        new_mask[indices] = max_value
        return new_mask
    
    def read_images(self, idx):
        p = self.p[idx]
        image = np.array(Image.open(self.dataset_root+p+'/Display.png').convert('RGB'))
        mask = np.array(Image.open(self.dataset_root+p+'/gt.png').convert('L')).astype(np.float32)
        maps = [np.array(Image.open(self.dataset_root+p+'/'+i).convert('RGB')) if not i=='CAGIInversedOutput.png' else np.array(Image.open(self.dataset_root+p+'/'+i).convert('RGB'))[:,:,::-1] for i in MAP_NAMES]
        data = {}
        data['image'] = image
        original_h,original_w = image.shape[:2]
        data['mask'] = mask
        data.update({map_name:maps[i] for i,map_name in enumerate(MAP_NAMES)})
        if self.transform is not None:      
            data = self.transform((original_h,original_w),self.image_size)(**data)
        data['mask'] = self.binarize_mask(data['mask'])
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
                    if(self.dct_flag and self.sb_flag):
                        dct = data.pop('DCTOutput.png')
                        sb = data.pop('SBOutput.png')
                        return image, mask, dct, sb
                    if self.dct_flag:
                        dct = data.pop('DCTOutput.png')
                        return image, mask, dct
                    if self.sb_flag:
                        sb = data.pop('SBOutput.png')
                        return image, mask, sb
                else:
                    return data
            except Exception as e:
                if self.verbose:
                    traceback.print_exc()
                idx=np.random.randint(0,len(self))
