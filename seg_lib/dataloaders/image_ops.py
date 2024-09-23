import cv2
import numpy as np

def resize_w_pad(img, size: tuple, interpolation: int = cv2.INTER_LINEAR):
    h, w = img.shape[:2]
    is_gs = len(img.shape) == 2
    c = img.shape[2] if not is_gs else 1

    if h == w: 
        return cv2.resize(img, size, interpolation)

    diff = h if h > w else w
    x_pos = (diff - w) // 2
    y_pos = (diff - h) // 2

    if is_gs:
        mask = np.zeros((diff, diff), dtype=img.dtype)
        mask[y_pos:y_pos+h, x_pos:x_pos+w] = img[:h, :w]
    else:
        mask = np.zeros((diff, diff, c), dtype=img.dtype)
        mask[y_pos:y_pos+h, x_pos:x_pos+w, :] = img[:h, :w, :]

    return cv2.resize(mask, size, interpolation)

def to_grayscale(image, return_three_channels: bool = True):
        '''Takes an RGB image and returns its equivallen in grayscale'''
        gs_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        if not return_three_channels:
            return gs_image

        image = np.zeros((*gs_image.shape, 3), dtype=np.uint8)
        image[:, :, 0] = gs_image
        image[:, :, 1] = gs_image
        image[:, :, 2] = gs_image
        del gs_image

        return image
