import numpy as np
import cv2

def gauss(img, sigma):
    row, col = img.shape
    mean = 0
    
    g = np.random.normal(mean, sigma, (row, col))
    g = g.reshape(row, col)
    noise = img + g
    return noise