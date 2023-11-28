import numpy as np
import cv2

def gaussian_reduce(original_img):
    blurred = cv2.GaussianBlur(original_img, (5,5), 2.5)
    
    new_rows = original_img.shape[0]//2
    
    new_cols = original_img.shape[1]//2
        
    reduced = np.zeros((new_rows, new_cols))
    kernel = np.array([1,4,6,4,1])/16
        
    for i in range(new_rows*2):
        for j in range(new_cols*2):
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