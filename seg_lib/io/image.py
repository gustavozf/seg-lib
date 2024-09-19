import base64
import io

import cv2
import numpy as np
from PIL import Image

def read_img(path:str) -> np.ndarray:
    return cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)

def read_bmask(path:str) -> np.ndarray:
    return cv2.imread(path, cv2.IMREAD_GRAYSCALE) / 255.0

def read_rmask(path:str) -> np.ndarray:
    return cv2.imread(path, cv2.IMREAD_UNCHANGED)

def img_data_to_pil(img_data):
    f = io.BytesIO()
    f.write(img_data)
    img_pil = Image.open(f)
    return img_pil

def img_data_to_arr(img_data):
    img_pil = img_data_to_pil(img_data)
    img_arr = np.array(img_pil)
    return img_arr

def img_b64_to_arr(img_b64):
    img_data = base64.b64decode(img_b64)
    img_arr = img_data_to_arr(img_data)
    return img_arr