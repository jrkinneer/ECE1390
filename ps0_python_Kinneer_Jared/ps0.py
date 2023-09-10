import cv2
import os

#absolute_path = os.path.join(os.getcwd(), 'ps0_python_Kinneer_Jared', 'output', 'ps0-1-a-1.png')
#2a color swap
img_1 = cv2.imread("./output/ps0-1-a-1.png")
#img_1 = cv2.imread(absolute_path)

cv2.imshow("test_window", img_1)
cv2.waitKey(0)
red_channel = img_1[:,:,0]
green_channel = img_1[:,:,1]

img_1[:,:,1] = red_channel
img_1[:,:,0] = green_channel

cv2.imshow("final_showing", img_1)
cv2.waitKey(0)

cv2.destroyAllWindows()