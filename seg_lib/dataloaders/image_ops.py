import cv2
import numpy as np

def get_pad_offsets(h: int, w: int):
    '''Returns the side of the padded square and the (x, y) offsets used by
       `resize_w_pad` to center an image of size (h, w).'''
    diff = max(h, w)
    return diff, (diff - w) // 2, (diff - h) // 2

def resize_w_pad(img, size: tuple, interpolation: int = cv2.INTER_LINEAR):
    h, w = img.shape[:2]
    is_gs = len(img.shape) == 2
    c = img.shape[2] if not is_gs else 1

    if h == w: 
        return cv2.resize(img, size, interpolation=interpolation)

    diff, x_pos, y_pos = get_pad_offsets(h, w)

    if is_gs:
        mask = np.zeros((diff, diff), dtype=img.dtype)
        mask[y_pos:y_pos+h, x_pos:x_pos+w] = img[:h, :w]
    else:
        mask = np.zeros((diff, diff, c), dtype=img.dtype)
        mask[y_pos:y_pos+h, x_pos:x_pos+w, :] = img[:h, :w, :]

    return cv2.resize(mask, size, interpolation=interpolation)

def unpad_resize(
        img, original_hw: tuple, interpolation: int = cv2.INTER_LINEAR):
    '''Inverse of `resize_w_pad`: maps a square image back to the original
       (h, w) size, removing the padding.'''
    h, w = int(original_hw[0]), int(original_hw[1])
    diff, x_pos, y_pos = get_pad_offsets(h, w)

    img = cv2.resize(img, (diff, diff), interpolation=interpolation)
    return img[y_pos:y_pos+h, x_pos:x_pos+w]

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
