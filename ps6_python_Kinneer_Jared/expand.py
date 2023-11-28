import numpy as np
import cv2

def expand(gaussian_img, original_shape):
        
    expanded = np.zeros((original_shape))
    
    kernel_odd = np.array([9,17,9])/32
    kernel_even = np.array([7,19,7])/32
    
    for i in range(expanded.shape[0]):
        for j in range(expanded.shape[1]):
            if i//2 == gaussian_img.shape[0]:
                old_row = i//2 - 1
            else:
                old_row = i//2
            
            if j%2 == 0:
                if j == 0:
                    expanded[i][j] = np.dot(kernel_even[1:], gaussian_img[old_row, j//2:j//2 + 2])
                elif j == gaussian_img.shape[1] * 2 - 2:
                    expanded[i][j] = np.dot(kernel_even[:2], gaussian_img[old_row, j//2 - 1: j//2 + 1])
                elif j == original_shape[1] - 1:
                    expanded[i][j] = np.dot(kernel_even[:1], gaussian_img[old_row, j//2 - 1: j//2 + 1])
                else:
                    expanded[i][j] = np.dot(kernel_even, gaussian_img[old_row, j//2 - 1:j//2 + 2])
            else:
                if j == 1:
                    expanded[i][j] = np.dot(kernel_odd[1:], gaussian_img[old_row, j//2:j//2 + 2])
                elif j == original_shape[1] - 2:
                    expanded[i][j] = np.dot(kernel_odd[:2], gaussian_img[old_row, j//2 - 1: j//2 + 1])
                elif j == original_shape[1] - 1:
                    expanded[i][j] = np.dot(kernel_odd[:2], gaussian_img[old_row, j//2 - 1: j//2 + 1])
                else:
                    expanded[i][j] = np.dot(kernel_odd, gaussian_img[old_row, j//2 - 1: j//2 + 2])
    return expanded