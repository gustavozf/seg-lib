import cv2
import numpy as np

def read_img(path:str) -> np.ndarray:
    return cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)

def read_bmask(path:str) -> np.ndarray:
    return cv2.imread(path, cv2.IMREAD_GRAYSCALE) / 255.0

def read_rmask(path:str) -> np.ndarray:
    return cv2.imread(path, cv2.IMREAD_UNCHANGED)