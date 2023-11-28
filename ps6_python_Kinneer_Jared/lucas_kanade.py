import numpy as np
import cv2
from scipy import ndimage

def lucas_kanade(img1, img2, window_size=5):
    u = np.zeros_like(img1)
    v = np.zeros_like(img1)
    
    #smooth imgaes before gradient calculation
    smoothed_img1 = cv2.GaussianBlur(img1, (5, 5), 1)
    smoothed_img2 = cv2.GaussianBlur(img2, (5, 5), 1)
    #calculate gradient
    kernel_x = np.array([[-1,1], [-1,1]])
    kernel_y = np.array([[-1,-1], [1,1]])
    grad_x = cv2.filter2D(smoothed_img1, -1, kernel_x)
    grad_y = cv2.filter2D(smoothed_img1, -1, kernel_y)

    kernel_t = np.array([[1,1],[1,1]])
    grad_t = cv2.filter2D(smoothed_img2, -1, kernel_t) - cv2.filter2D(smoothed_img1, -1, kernel_t)
    
    w = window_size//2
    
    for i in range(w, img1.shape[0] - w):
        for j in range(w, img1.shape[1] - w):
            
            #for window size 5 flattened arrays will be of size (25, )
            #or generalized to be of size (window_size**2, )
            Ix = grad_x[i-w : i+w+1, j-w : j+w+1].flatten()
            Iy = grad_y[i-w : i+w+1, j-w : j+w+1].flatten()
            It = grad_t[i-w : i+w+1, j-w : j+w+1].flatten()
            
            #forces the It array to be of size (window_size**2, 1) 
            b = np.reshape(It, (It.shape[0], 1))
            #stacks the gradient_x and gradient_y window results to into an array of size (window_size**2, 2)
            A = np.vstack((Ix, Iy)).T
            
            #gets result of 
            #[sum(Ix*Ix), sum(Ix*Iy)] [u]   -[sum(Ix*It)]
            #[sum(Ix*Iy), sum(Iy*Iy)] [v] =  [sum(Iy*It)]
            #==
            #   A * d = b
            # d = A^-1*b
            d = np.matmul(np.linalg.pinv(A), b)
            
            u[i][j] = d[0][0]
            v[i][j] = d[1][0]
    
    return u,v