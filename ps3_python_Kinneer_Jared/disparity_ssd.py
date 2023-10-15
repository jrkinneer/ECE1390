import numpy as np
from tqdm import *

BLOCK_DIM = 7
SEARCH_BLOCK_SIZE = BLOCK_DIM * 12

def disparity_ssd(L, R):
    """Compute disparity map D(y, x) such that: L(y, x) = R(y, x + D(y, x))
    
    Params:
    L: Grayscale left image
    R: Grayscale right image, same size as L

    Returns: Disparity map, same size as L, R
    """
    h, w = L.shape
    disparity_map = np.zeros((h,w))
    for y in tqdm(range(BLOCK_DIM, h - BLOCK_DIM)):
        for x in range(BLOCK_DIM, w-BLOCK_DIM):
            block_left = np.asarray(L[y:y+BLOCK_DIM, x:x+BLOCK_DIM])
            
            min_ind = compare_along_line(y, x, R, block_left, block_size=BLOCK_DIM)
            
            disparity_map[y, x] = abs(min_ind[1] - x)
    
    return disparity_map
    # TODO: Your code here

def SSD_per_block(left_block, right_block):
    return np.sum(np.power(np.subtract(left_block, right_block), 2))

def compare_along_line(row_ind, col_ind, right_img, left_block, block_size=7):
    x_left = max(0, col_ind - SEARCH_BLOCK_SIZE)
    x_right = min(right_img.shape[1] - block_size, col_ind + SEARCH_BLOCK_SIZE + block_size)
    first = True
    min_SSD = None
    min_ind = None
    
    for x in range(x_left, x_right):
        right_block = np.asarray(right_img[row_ind: row_ind+block_size, x: x+block_size])
        
        if (left_block.shape != right_block.shape):
            print("\n shape: ", right_img.shape[1] - 1, " other: ", col_ind + SEARCH_BLOCK_SIZE + 1)
            print("row_ind: ", row_ind, " col_ind: ", col_ind)
        
        ssd = SSD_per_block(left_block, right_block)
        
        if first:
            min_SSD = ssd
            min_ind = (row_ind,x)
            first = False
        else:
            if (ssd < min_SSD):
                min_SSD = ssd
                min_ind = (row_ind, x)
                
    return min_ind