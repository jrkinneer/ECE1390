# ps2
import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from disparity_ssd import disparity_ssd
from gauss import gauss

## 1-a
# Read images
# L = cv2.imread(os.path.join('input', 'pair0-L.png'), 0) * (1.0 / 255.0)  # grayscale, [0, 1]
# R = cv2.imread(os.path.join('input', 'pair0-R.png'), 0) * (1.0 / 255.0)

# Compute disparity (using method disparity_ssd defined in disparity_ssd.py)
# D_L = disparity_ssd(L, R)
# cv2.imshow("first disparity", D_L)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
# D_R = disparity_ssd(R, L)
# cv2.imshow("second disparity", D_R)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
# cv2.imwrite("./output/ps3-1-a-1.png", D_L)
# cv2.imwrite("./output/ps3-1-a-2.png", D_R)


# TODO: Save output images (D_L as output/ps2-1-a-1.png and D_R as output/ps2-1-a-2.png)
# Note: They may need to be scaled/shifted before saving to show results properly

# TODO: Rest of your code here

#2

# L1 = cv2.imread(os.path.join('input', 'pair1-L.png'), cv2.IMREAD_GRAYSCALE) * (1.0 / 255.0)  # grayscale, [0, 1]
# R1 = cv2.imread(os.path.join('input', 'pair1-R.png'), cv2.IMREAD_GRAYSCALE) * (1.0 / 255.0)
# cv2.imshow("show L1", L1)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
# D_L = disparity_ssd(L1, R1)
# cv2.imwrite("./output/ps3-2-a-1.png", D_L)
# D_R = disparity_ssd(R1, L1)
# cv2.imwrite("./output/ps3-2-a-2.png", D_R)

#3
# L1 = cv2.imread(os.path.join('input', 'pair1-L.png'), cv2.IMREAD_GRAYSCALE) * (1.0 / 255.0)  # grayscale, [0, 1]
# R1 = cv2.imread(os.path.join('input', 'pair1-R.png'), cv2.IMREAD_GRAYSCALE) * (1.0 / 255.0)
# noisy_L1 = gauss(L1, .1)
# noisy_R1 = gauss(R1, .1)
# D_L = disparity_ssd(noisy_L1, noisy_R1)
# cv2.imwrite("./output/ps3-3-a-1.png", D_L)
# D_R = disparity_ssd(noisy_R1, noisy_L1)
# cv2.imwrite("./output/ps3-3-a-2.png", D_R)


#3b
L1 = cv2.imread(os.path.join('input', 'pair1-L.png'), cv2.IMREAD_GRAYSCALE) * (1.0 / 255.0)  # grayscale, [0, 1]
R1 = cv2.imread(os.path.join('input', 'pair1-R.png'), cv2.IMREAD_GRAYSCALE) * (1.0 / 255.0)
mult_R1 = R1 * 1.1
D_L = disparity_ssd(L1, mult_R1)
cv2.imwrite("./output/ps3-3-b-1.png", D_L)
D_R = disparity_ssd(mult_R1, L1)
cv2.imwrite("./output/ps3-3-b-2.png", D_R)
#4