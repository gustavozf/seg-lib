import logging
from typing import Tuple

import cv2
import numpy as np
from scipy import ndimage

class Sampler:
    def __init__(
            self,
            sampling_step: int = 50, min_blob_count: int = 10,
            mode: str = 'grid', erode_grid: str = 'on'):
        self.sampling_step  = sampling_step
        self.min_blob_count = min_blob_count
        self.mode = mode
        self.erode_grid = erode_grid
    
    def sample_pixels(
            self,
            mask_of_blobs: np.ndarray,
            mask: np.ndarray
        ) -> Tuple[np.ndarray, np.ndarray]:
        # draw a pix for each blob
        input_point, input_label = [], []
        blob_labels, blob_sample = np.unique(mask_of_blobs, return_index=True)
        gt_fl = mask.flatten()
        for bl, bs in zip(blob_labels, blob_sample):
            mask_bool = (mask_of_blobs==bl)
            count = mask_bool.sum()
            ## it's not a background blob or a false blob
            if gt_fl[bs]>=1.0 and count>self.min_blob_count:
                x_center, y_center = np.argwhere(mask_bool).sum(0)/count
                x_center = int(x_center) % mask.shape[0]
                y_center = int(y_center) % mask.shape[1]
                input_point.append([y_center, x_center])
                input_label.append(1)

                logging.debug(
                    f"blob #{bl} drawn point: {[x_center, y_center]}"
                )

        # no mask? pick the center pixel of image
        if len(input_point) == 0:
            input_point = [[mask.shape[1]//2, mask.shape[0]//2]]
            input_label = [1]

        return np.array(input_point), np.array(input_label)
    
    def sample_pixels_center_of_mass(
            self,
            mask_of_blobs: np.ndarray,
            mask: np.ndarray
        ) -> Tuple[np.ndarray, np.ndarray]:
        # draw a pix for each blob
        input_point, input_label = [], []
        blob_labels, blob_sample = np.unique(mask_of_blobs, return_index=True)
        gt_fl = mask.flatten()
        for bl, bs in zip(blob_labels, blob_sample):
            mask_bool = (mask_of_blobs==bl)
            count = mask_bool.sum()
            ## it's not a background blob or a false blob
            if gt_fl[bs]>=1.0 and count>self.min_blob_count:
                x_center, y_center = ndimage.center_of_mass(mask_bool)
                input_point.append([y_center, x_center])
                input_label.append(1)
                
                logging.debug(
                    f"blob #{bl} drawn point: {[x_center, y_center]}"
                )

        # no mask? pick the center pixel of image
        if len(input_point) == 0:
            input_point = [[mask.shape[1]//2, mask.shape[0]//2]]
            input_label = [1]
            logging.debug(f"empty blob -> {input_point}")

        return np.array(input_point), np.array(input_label)

    def sample_pixels_random(
            self, mask_of_blobs: np.ndarray, mask: np.ndarray
        ) -> Tuple[np.ndarray, np.ndarray]:
        # draw a pix for each blob
        input_point, input_label = [], []
        blob_labels, blob_sample = np.unique(mask_of_blobs, return_index=True)
        gt_fl = mask.flatten()
        for bl, bs in zip(blob_labels, blob_sample):
            mask_bool = (mask_of_blobs==bl)
            count = mask_bool.sum()
            ## it's not a background blob or a false blob
            if gt_fl[bs]>=1.0 and count>self.min_blob_count:
                indices = np.argwhere(mask_bool)
                random_index = np.random.choice(indices.shape[0])
                x_center, y_center = indices[random_index]
                input_point.append([y_center, x_center])
                input_label.append(1)
                
                logging.debug(
                    f"blob #{bl} drawn point: {[x_center, y_center]}"
                )

        # no mask? sample a random point
        if len(input_point) == 0:
            input_point, input_label = [ \
                [np.random.randint(0, mask.shape[1]),   \
                 np.random.randint(0, mask.shape[0])]], \
            [1]

        return np.array(input_point), np.array(input_label)

    def get_grid(self, mask, offset_px_x, offset_px_y):
        row = np.zeros(mask.shape, dtype=int)
        col = np.zeros(mask.shape, dtype=int)

        for i in range(offset_px_y, row.shape[0], self.sampling_step):
            row[i, :] = 1
        for i in range(offset_px_x, col.shape[1], self.sampling_step):
            col[:, i] = 1
        res = row & col
        return res
    
    def sample_pixels_grid(
            self, mask_of_blobs: np.ndarray, mask: np.ndarray
        ) -> Tuple[np.ndarray, np.ndarray]:
        # draw a pix for each blob
        input_point = []
        offset_px_x = 0
        offset_px_y = 0
        while len(input_point)==0 and offset_px_y < self.sampling_step:
            res = self.get_grid(mask, offset_px_x, offset_px_y)

            input_point = np.argwhere(res & mask.astype(np.int64))
            input_point[:, (0, 1)] = input_point[:, (1, 0)]
            
            offset_px_x += 1
            if offset_px_x>self.sampling_step:
                offset_px_x = 0
                offset_px_y += 1

        # STILL no mask? sample a random point
        blob_labels = np.unique(mask_of_blobs)
        if len(input_point) <= blob_labels.shape[0]-1:
            return self.sample_pixels_random(mask_of_blobs, mask)

        input_label = [1 for _ in input_point]
        return np.array(input_point), np.array(input_label)
    
    def sample_pixels_eroded_grid(
            self, mask_of_blobs: np.ndarray, mask: np.ndarray
        ) -> Tuple[np.ndarray, np.ndarray]:
        input_point = []
        offset_px_x = 0
        offset_px_y = 0
        
        while len(input_point)==0 and offset_px_y < self.sampling_step:
            res = self.get_grid(mask, offset_px_x, offset_px_y)
            erode_size = 10
        
            while True:
                # Erode the mask
                kernel = cv2.getStructuringElement(
                    cv2.MORPH_ELLIPSE, (erode_size, erode_size)
                )
                eroded_mask = cv2.erode(mask, kernel)

                input_point = np.argwhere(res & eroded_mask.astype(np.int64))
                input_point[:, (0, 1)] = input_point[:, (1, 0)]

                blob_labels, blob_sample = np.unique(
                    mask_of_blobs, return_index=True
                )
                gt_fl = mask.flatten()

                blobs = np.zeros(blob_labels.shape[0], dtype=np.float32)
                for i, (bl, bs) in enumerate(zip(blob_labels, blob_sample)):
                    mask_bool = (mask_of_blobs==bl)
                    count = mask_bool.sum()
                    ## it's not a background blob or a false blob
                    if not (gt_fl[bs]>=1.0 and count>self.min_blob_count):
                        blobs[i] = -1

                for i in range(input_point.shape[0]):
                    fl_ip = input_point[i][0]*mask.shape[1] + input_point[i][1]
                    idx = mask_of_blobs[input_point[i][1], input_point[i][0]]
                    blobs[idx]=1.0
                
                if not np.any(blobs==0.0) or erode_size==1:
                    break
                erode_size -= 1
            
            offset_px_x += 1
            if offset_px_x>self.sampling_step:
                offset_px_x = 0
                offset_px_y += 1
            
        
        input_label = [1 for _ in input_point]

        # still no mask? sample a random point
        if len(input_point) == 0:
            return self.sample_pixels_grid(mask_of_blobs, mask)

        return np.array(input_point), np.array(input_label)
    
    def sample(self, mask_of_blobs: np.ndarray, mask: np.ndarray):
        if self.mode in ('common', 'A'):
            return self.sample_pixels(mask_of_blobs, mask)
        elif self.mode in ('center_of_mass', 'B'):
            return self.sample_pixels_center_of_mass(mask_of_blobs, mask)
        elif self.mode in ('random', 'C'):
            return self.sample_pixels_random(mask_of_blobs, mask)
        elif self.mode in ('grid', 'D'):
            if self.erode_grid=="on":
                return self.sample_pixels_eroded_grid(mask_of_blobs, mask)
            else:
                return self.sample_pixels_grid(mask_of_blobs, mask)
