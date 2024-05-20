import albumentations as abm
import cv2

HIGH_P = 0.4
MED_P = 0.2
SMALL_P = 0.1

class BaseAugmenter(object):
    # Equivallent to DA1, presented in:
    #   Nanni L, Cuza D, Lumini A, Loreggia A, Brahnam S (2021) 
    #   Deep ensembles in bioimage segmentation.
    #   arXiv preprint arXiv:211212955
    def __init__(self):
        self.transformations = abm.OneOf([
            abm.HorizontalFlip(p=1.0),
            abm.VerticalFlip(p=1.0),
            abm.Rotate((-90, 90), border_mode=cv2.BORDER_CONSTANT, p=1.0)
        ], p=1.0)

    def __call__(self, image, mask):
        augmented = self.transformations(image=image, mask=mask)
        return augmented['image'], augmented['mask']

class WeakImgSegAugmenter(BaseAugmenter):
    def __init__(self, img_size: int = 256):
        self.transformations = abm.Compose([
            abm.HorizontalFlip(p=HIGH_P),
            abm.Rotate((-30, 30), border_mode=cv2.BORDER_CONSTANT, p=HIGH_P),
            abm.Affine(p=MED_P),
            abm.RandomBrightnessContrast(p=MED_P),
            abm.ColorJitter(p=MED_P),
            abm.RandomSizedCrop(
                min_max_height=(int(img_size * 0.64), int(img_size * 0.8)),
                height=img_size, width=img_size,
                p=SMALL_P),
            abm.Downscale(
                scale_min=0.6, scale_max=0.8,
                interpolation=cv2.INTER_AREA,
                p=SMALL_P)
        ])

class NormalImgSegAugmenter(BaseAugmenter):
    def __init__(self, img_size: int = 256):
        self.transformations = abm.Compose([
            abm.VerticalFlip(p=HIGH_P),
            abm.HorizontalFlip(p=HIGH_P),
            abm.Rotate((-30, 30), border_mode=cv2.BORDER_CONSTANT, p=HIGH_P),
            abm.OneOf([
                abm.GaussianBlur(p=MED_P),
                abm.MedianBlur(p=HIGH_P),
                abm.MotionBlur(p=HIGH_P)
            ], p=MED_P),
            abm.Affine(p=MED_P),
            abm.RandomBrightnessContrast(p=MED_P),
            abm.ColorJitter(p=MED_P),
            abm.RandomSizedCrop(
                min_max_height=(int(img_size * 0.64), int(img_size * 0.8)),
                height=img_size, width=img_size,
                p=SMALL_P),
            abm.Downscale(
                scale_min=0.6, scale_max=0.8,
                interpolation=cv2.INTER_AREA,
                p=SMALL_P)
        ])

aug_repository = {
    'base': BaseAugmenter,
    'weak': WeakImgSegAugmenter,
    'default': NormalImgSegAugmenter
}