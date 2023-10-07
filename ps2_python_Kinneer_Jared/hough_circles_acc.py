import numpy as np
import cv2

def hough_circles_acc(img_edges, radius):
    #calculate diagonal size of the image, rho, and theta
    height, width = img_edges.shape
    diagonal = np.ceil(np.sqrt(height**2 + width**2))
    rhos = np.arange(-diagonal, diagonal + 1, 1)
    theta_vals = np.deg2rad(np.arange(-90, 90, 1))
    
    #initialize empty H accumulator array
    H = np.zeros((len(rhos), len(theta_vals), radius), dtype=np.uint64)
    
    y_idx, x_idx = np.nonzero(img_edges)
    
    for i in range(len(x_idx)):
        x = x_idx[i]
        y = y_idx[i]
        
        for k in range(radius):
            for j in range(len(theta_vals)):
                a = min(int(x - (k*np.cos(theta_vals[j]))), len(rhos) - 1)
                b = min(int(y + (k*np.sin(theta_vals[j]))), len(theta_vals) - 1)
                H[a][b][k] += 1
            
    return H