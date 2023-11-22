import cv2
import numpy as np
from lucas_kanade import lucas_kanade as LK

#lucas kanade optic flow
origin_image = cv2.imread("./input/TestSeq/Shift0.png", cv2.IMREAD_GRAYSCALE)
second_image = cv2.imread("./input/TestSeq/ShiftR2.png", cv2.IMREAD_GRAYSCALE)

u,v = LK(origin_image, second_image, 3)
cv2.imshow("u", u)
cv2.waitKey(0)
cv2.destroyAllWindows()

cv2.imshow("v", v)
cv2.waitKey(0)
cv2.destroyAllWindows()

cv2.imshow("u**2 + v**2", u**2 + v**2)
cv2.waitKey(0)
cv2.destroyAllWindows()
# u = np.zeros_like()
# for i in range(origin_image.shape[0]):
#     for j in range(origin_image.shape[1]):
        