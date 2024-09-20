import os

import cv2
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset

from seg_lib.dataloaders.data_aug import BaseAugmenter
from seg_lib.dataloaders.image_ops import resize_w_pad, to_grayscale
from seg_lib.io.image import read_img as read_img_cv


class SegGeneralDataset(Dataset):
    """
    Reads the images and applies the augmentation transform on them.
    """

    def __init__(
            self, dataset_path: str,
            df_file_path: str = 'metadata/dataset.csv',
            split: str = 'train', img_size: int = 256,
            read_img_as_grayscale: bool = False, 
            pixel_mean: list[float] = [123.675, 116.28, 103.53],
            pixel_std: list[float] = [58.395, 57.12, 57.375],
            augmenter: BaseAugmenter = None) -> None:
        self.dataset_path = dataset_path
        self.split = split
        self.img_size = img_size
        self.read_img_as_grayscale = read_img_as_grayscale
        self.augmenter = augmenter
        # default values obtained from SAM
        self.pixel_mean = torch.Tensor(pixel_mean).view(-1, 1, 1)
        self.pixel_std = torch.Tensor(pixel_std).view(-1, 1, 1)
        # metadata dataframe
        self.df = pd.read_csv(os.path.join(dataset_path, df_file_path))
        self.df = self.df[self.df['split'] == split]

    def __len__(self):
        return self.df.shape[0]

    def get_num_classes(self, subset: str):
        return self.df[self.df['subset'] == subset]['class_id'].max() + 1

    def read_img(self, img_name: str, label_name: str, subset: str):
        base_path = os.path.join(self.dataset_path, subset)
        img_path = os.path.join(base_path, 'img', img_name)
        label_path = os.path.join(base_path, 'label', label_name)

        if not os.path.exists(img_path):
            raise ValueError(f'Image does not exist on disk: {img_path}')
        if not os.path.exists(label_path):
            raise ValueError(f'Label does not exist on disk: {label_path}')

        # load the image (H, W, 3) and convert it from BGR to RGB
        image = read_img_cv(img_path)
        # read the mask as (H, W), grayscale
        mask = cv2.imread(label_path, 0)

        if self.read_img_as_grayscale:
            image = to_grayscale(image)

        if self.get_num_classes(subset) == 2:
            mask[mask > 1] = 1

        return image, mask
    
    def resize(self, image, mask):
        ''' Resize the inputs and create the low-level mask (used for measuring
            the loss).
        '''
        img_size = (self.img_size, self.img_size)
        image = resize_w_pad(image, img_size, interpolation=cv2.INTER_LINEAR)
        mask = resize_w_pad(mask, img_size, interpolation=cv2.INTER_NEAREST)

        return image.astype(np.uint8), mask.astype(np.uint8)
            
    def __getitem__(self, i):
        row = dict(self.df.iloc[i])
        image, mask = self.read_img(
            row['img_name'], row['label_name'], row['subset'])
        original_img_size = image.shape[:2]

        # resize the images
        image, mask = self.resize(image, mask)
        # apply data aug
        if self.augmenter is not None:
            image, mask = self.augmenter(image, mask)

        # fix the masks to use only the targe class id
        mask[mask != row['class_id']] = 0
        mask[mask == row['class_id']] = 1

        # put the image in the torch format and normalize it
        image = torch.from_numpy(image).permute(2, 0, 1).contiguous()
        image = (image - self.pixel_mean) / self.pixel_std

        return {
            'image': image,
            'label': torch.from_numpy(mask[np.newaxis, ...]).long(),
            'original_img_size': original_img_size,
            **row
        }
