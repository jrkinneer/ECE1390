import numpy as np
import cv2

def gaussian_reduce(original_img):
    blurred = cv2.GaussianBlur(original_img, (5,5), 2.5)
    reduced = np.zeros_like(original_img[1::2, 1::2])
    kernel = np.array([1,4,6,4,1])/16
    
    for i in range(original_img.shape[0]):
        for j in range(original_img.shape[1]):
            if i%2 == 1 and j%2==1:
                if j == 1:
                    window=blurred[i][j:j+3]
                    m = kernel[2:] * window
                elif j == original_img.shape[1] - 2:
                    window=blurred[i][j-3:j]
                    m = kernel[:3] * window
                elif j > 1 and j < original_img.shape[1] - 2:
                    window = blurred[i][j-2:j+3]
                    m = kernel * window
                    
                reduced[i//2][j//2] = sum(m)

    return reduced