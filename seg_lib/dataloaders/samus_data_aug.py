"""
MIT License

Copyright (c) 2023 Xian Lin

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
===============================================================================
Disclaimer: the here presented script was adapted by the repository owners.
The original version may be found at https://github.com/xianlin7/SAMUS
"""

import numpy as np
from torchvision.transforms import functional as F
from torchvision.transforms import (
    InterpolationMode,
    ColorJitter,
    RandomAffine,
    RandomCrop,
    RandomRotation
)

class SamusImgAugmenter(object):
    def __init__(
            self,
            img_size=256, crop=(32, 32),
            p_flip=0.0, p_rota=0.0, p_scale=0.0, p_gaussn=0.0,
            p_contr=0.0, p_gama=0.0, p_distor=0.0, p_random_affine=0.0,
            color_jitter_params=(0.1, 0.1, 0.1, 0.1)):
        self.crop = crop
        self.p_flip = p_flip
        self.p_rota = p_rota
        self.p_scale = p_scale
        self.p_gaussn = p_gaussn
        self.p_gama = p_gama
        self.p_contr = p_contr
        self.p_distortion = p_distor
        self.img_size = img_size
        self.color_jitter_params = color_jitter_params
        self.p_random_affine = p_random_affine
        self.color_tf = (
            ColorJitter(*color_jitter_params) if color_jitter_params
            else lambda img: img
        )
    
    @staticmethod
    def init_eval(img_size=256):
        return SamusImgAugmenter(
            img_size=img_size, crop=None, color_jitter_params=None)

    @staticmethod
    def init_train(img_size=256):
        return SamusImgAugmenter(
            img_size=img_size, crop=None, color_jitter_params=None,
            p_rota=0.5, p_scale=0.5, p_contr=0.5, p_distor=0.5)

    # ---------------------------------------------------------------- AUGMENTS
    def random_gamma(self, image):
        if np.random.rand() < self.p_gama:
            c = 1
            g = np.random.randint(10, 25) / 10.0
            image = (np.power(image / 255, 1.0 / g) / c) * 255
            image = image.astype(np.uint8)
        return image

    def random_crop(self, image, mask):
        if self.crop:
            i, j, h, w = RandomCrop.get_params(image, self.crop)
            image, mask = F.crop(image, i, j, h, w), F.crop(mask, i, j, h, w)
        return image, mask
    
    def horizontal_flip(self, image, mask):
        if np.random.rand() < self.p_flip:
            image, mask = F.hflip(image), F.hflip(mask)
        return image, mask

    def random_rotation(self, image, mask):
        if np.random.rand() < self.p_rota:
            angle = RandomRotation.get_params((-30, 30))
            image, mask = F.rotate(image, angle), F.rotate(mask, angle)
        return image, mask
    
    def random_scale(self, image, mask):
        if np.random.rand() < self.p_scale:
            scale = np.random.uniform(1, 1.3)
            new_shape = int(self.img_size * scale), int(self.img_size * scale)
            image, mask = (
                F.resize(image, new_shape, InterpolationMode.BILINEAR),
                F.resize(mask, new_shape, InterpolationMode.NEAREST)
            )
            i, j, h, w = RandomCrop.get_params(
                image, (self.img_size, self.img_size)
            )
            image, mask = F.crop(image, i, j, h, w), F.crop(mask, i, j, h, w)
        return image, mask
    
    def random_gauss_noise(self, image):
        if np.random.rand() < self.p_gaussn:
            ns = np.random.randint(3, 15)
            noise = np.random.normal(
                loc=0, scale=1, size=(self.img_size, self.img_size)
            ) * ns
            noise = noise.astype(int)
            image = np.array(image) + noise
            image[image > 255] = 255
            image[image < 0] = 0
            image = F.to_pil_image(image.astype('uint8'))
        return image
    
    def random_contrast(self, image):
        if np.random.rand() < self.p_contr:
            contr_tf = ColorJitter(contrast=(0.8, 2.0))
            image = contr_tf(image)
        return image
        
    def random_distortion(self, image):
        if np.random.rand() < self.p_distortion:
            distortion = RandomAffine(0, None, None, (5, 30))
            image = distortion(image)
        return image
    
    def random_affine_transform(self, image, mask):
        if np.random.rand() < self.p_random_affine:
            affine_params = RandomAffine(180).get_params(
                (-90, 90), (1, 1), (2, 2), (-45, 45), self.crop
            )
            image = F.affine(image, *affine_params)
            mask = F.affine(mask, *affine_params)
        return image, mask
    ## --------------------------------------------------------- APPLY AUGMENTS
    def __call__(self, image, mask):
        #  gamma enhancement
        image = self.random_gamma(image)
        # transforming to PIL image
        image, mask = F.to_pil_image(image), F.to_pil_image(mask)
        # random crop
        image, mask = self.random_crop(image, mask)
        # random horizontal flip
        image, mask = self.horizontal_flip(image, mask)
        # random rotation
        image, mask = self.random_rotation(image, mask)
        # random scale and center resize to the original size
        image, mask = self.random_scale(image, mask)
        # random add gaussian noise
        image = self.random_gauss_noise(image)
        # random change the contrast
        image = self.random_contrast(image)
        # random distortion
        image = self.random_distortion(image)
        # color transforms || ONLY ON IMAGE
        image = self.color_tf(image)
        # random affine transform
        image, mask = self.random_affine_transform(image, mask)

        return image, mask