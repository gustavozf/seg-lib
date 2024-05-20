import os

import cv2
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset

from seg_lib.dataloaders.data_aug import BaseAugmenter
from seg_lib.prompt.train_sampler import TrainPromptSampler
from seg_lib.prompt.samus_sampler import random_bbox, fixed_bbox
from seg_lib.dataloaders.seg_dataset import (
    SegGeneralDataset, resize_w_pad
)

DEFAULT_SAMPLER = TrainPromptSampler(min_blob_count=10, max_num_prompts=1)

class SamGeneralDataset(SegGeneralDataset):
    """
    Reads the images, applies the augmentation and generate the input prompts.
    """

    def __init__(
            self, dataset_path: str,
            df_file_path: str = 'metadata/dataset.csv',
            split: str = 'train', prompt: str = 'click',
            img_size: int = 256, embedding_size: int = 128,
            read_img_as_grayscale: bool = False, 
            pixel_mean: list[float] = [123.675, 116.28, 103.53],
            pixel_std: list[float] = [58.395, 57.12, 57.375],
            point_sampler: TrainPromptSampler = DEFAULT_SAMPLER,
            augmenter: BaseAugmenter = None) -> None:
        super().__init__(
            dataset_path,
            df_file_path=df_file_path,
            split=split,
            img_size=img_size,
            read_img_as_grayscale=read_img_as_grayscale, 
            pixel_mean=pixel_mean,
            pixel_std=pixel_std,
            augmenter=augmenter)
        self.prompt = prompt
        self.embedding_size = embedding_size
        # point prompt sampler
        self.point_sampler = point_sampler

    def get_prompt(self, mask, class_id):
        '''Get the prompts from the label mask'''      
        prompts = {}
        
        if self.prompt in {'bbox', 'all'}:
            prompts['boxes'] = (
                random_bbox(mask, class_id, self.img_size)
                if 'train' in self.split
                else fixed_bbox(mask, class_id, self.img_size)
            )
        
        if self.prompt in {'click', 'all'}:
            pt, point_label = self.point_sampler(
                mask, class_id=class_id,
                index_sampler='random' if 'train' in self.split else 'fixed'
            )
            prompts['point_coords'] = torch.as_tensor(pt, dtype=torch.float32)
            prompts['point_labels'] = torch.as_tensor(
                point_label, dtype=torch.int
            )

        return prompts
            
    def __getitem__(self, i):
        row = dict(self.df.iloc[i])
        image, mask = self.read_img(
            row['img_name'], row['label_name'], row['subset']))
        original_img_size = image.shape[:2]

        # resize the images
        image, mask = self.resize(image, mask)
        # apply data aug
        if self.augmenter is not None:
            image, mask = self.augmenter(image, mask)

        # generate the prompts from teh mask
        prompts = self.get_prompt(mask, row['class_id'])
        # fix the masks to use only the targe class id
        mask[mask != row['class_id']] = 0
        mask[mask == row['class_id']] = 1

        # put the image in the torch format and normalize it
        image = torch.from_numpy(image).permute(2, 0, 1).contiguous()
        image = (image - self.pixel_mean) / self.pixel_std

        # create the low level mask (used for loss computing)
        emb_size = (self.embedding_size, self.embedding_size)
        low_mask = resize_w_pad(
            mask, emb_size, interpolation=cv2.INTER_NEAREST
        ).astype(np.uint8)

        return {
            'image': image,
            'label': torch.from_numpy(mask[np.newaxis, ...]).long(),
            'low_mask': torch.from_numpy(low_mask[np.newaxis, ...]).long() ,
            'original_img_size': original_img_size,
            **prompts,
            **row
        }
