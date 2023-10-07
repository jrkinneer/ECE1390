import numpy as np

def hough_lines_acc(img_edges, theta=(-90,90)):
    """_summary_

    Args:
        img_edges (img): image as np array
        theta (int, int): tuple of the start and end point of range of theta, theta[0] must be less than theta[1]

    Returns:
        tuple: returns tuple of Hough cummulator array ('uint8'), theta and rho values for that array
    """
    #calculate diagonal size of the image, rho, and theta
    height, width = img_edges.shape
    diagonal = np.ceil(np.sqrt(height**2 + width**2))
    rhos = np.arange(-diagonal, diagonal + 1, 1)
    theta_vals = np.deg2rad(np.arange(theta[0], theta[1], 1))
    
    #initialize empty H accumulator array
    H = np.zeros((len(rhos), len(theta_vals)), dtype=np.uint64)
    
    #finds the edge indices in img_edges
    y_idx, x_idx = np.nonzero(img_edges)
    
    for i in range((len(x_idx))):
        x = x_idx[i]
        y = y_idx[i]
        for j in range(len(theta_vals)):
            rho = int((x * np.cos(theta_vals[j])) + (y * np.sin(theta_vals[j])) + diagonal)
            H[rho, j] += 1
    
    return (H.astype('uint8'), theta_vals, rhos)