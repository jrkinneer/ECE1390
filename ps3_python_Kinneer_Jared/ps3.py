import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from disparity_ssd import disparity_ssd

L1 = cv2.imread(os.path.join('input', 'pair1-L.png'), cv2.IMREAD_GRAYSCALE) * (1.0 / 255.0)  # grayscale, [0, 1]
R1 = cv2.imread(os.path.join('input', 'pair1-R.png'), cv2.IMREAD_GRAYSCALE) * (1.0 / 255.0)
cv2.imshow("show L1", L1)

D_L = disparity_ssd(L1, R1)
cv2.imwrite("./output/ps3-2-a-1_test_color.png", D_L)
D_R = disparity_ssd(R1, L1)
cv2.imwrite("./output/ps3-2-a-2_test_color.png", D_R)