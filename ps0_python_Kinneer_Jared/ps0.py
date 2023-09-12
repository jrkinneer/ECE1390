import cv2
import numpy as np
import matplotlib.pyplot as plt 

#2a color swap
img_1 = cv2.imread("./output/ps0-1-a-1.png")
img1_copy = img_1
red_channel = img_1[:,:,2].copy()
green_channel = img_1[:,:,1].copy()
blue_channel = img_1[:,:,0].copy()

img1_copy[:,:,1] = red_channel
img1_copy[:,:,2] = green_channel

cv2.imwrite("./output/ps0-2-a-1.png", img1_copy)

#2b
img1_green = green_channel
cv2.imwrite("./output/ps0-2-b-1.png", img1_green)

#2c
img1_red = red_channel
cv2.imwrite("./output/ps0-2-c-1.png", img1_red)

#3a
img_2 = cv2.imread("./output/ps0-1-a-2.png")
img_2_red = img_2[:,:,2].copy()

img1_dimensions = img_1.shape
img1_height = img1_dimensions[0]
img1_width = img1_dimensions[1]

img2_dimensions = img_2.shape
img2_height = img2_dimensions[0]
img2_width = img2_dimensions[1]

start_range_x = int(img1_width/2 - 50)
stop_range_x = int((img1_width/2 - 50) + 100)
start_range_y = int(img1_height/2 - 50)
stop_range_y = int((img1_height/2 - 50) + 100)

##isolate 100*100 square of grand canyon, in green
square = red_channel[start_range_y:stop_range_y, start_range_x: stop_range_x]
#replace square in red channel of Cathy with grandcanyon square
start_range_x2 = int(img2_width/2 - 50)
stop_range_x2 = int((img2_width/2 - 50) + 100)
start_range_y2 = int(img2_height/2 - 50)
stop_range_y2 = int((img2_height/2 - 50) + 100)
img_2_red[start_range_y2:stop_range_y2, start_range_x2: stop_range_x2] = square
#merge three channels of image two
img2_recreated = cv2.merge((img_2[:,:,0], img_2[:,:,1], img_2_red))
cv2.imwrite("./output/ps0-3-a-1.png", img2_recreated)

#3b
square_white = np.full((100,100,3), 255)
img2_white_square = img_2
img2_white_square[start_range_y2:stop_range_y2, start_range_x2: stop_range_x2] = square_white
cv2.imwrite("./output/ps0-3-b-1.png", img2_white_square)

#4a find max of img1 green channel
minVal_img1_green = np.amin(img1_green)
maxVal_img1_green = np.amax(img1_green)
print('min: ', minVal_img1_green, ' max: ' , maxVal_img1_green)

#4b histogram of green channel
histG = cv2.calcHist([img_1], [1], None, [256], [0,256])
# plt.plot(histG)
# plt.show()

#4c mean and std dev operations
mean, stddev = cv2.meanStdDev(img_1)
print('mean: ', mean, ' std dev: ', stddev)
#apply mean and std operations to green channel
altered_green = img_1[:,:,1] - mean[1]
altered_green = altered_green/stddev[1]
altered_green = altered_green*10
altered_green = altered_green + mean[1]


cv2.imwrite("./output/ps0-4-c-1.png", altered_green)
#4d histogram of altered image
altered_green_png = cv2.imread("./output/ps0-4-c-1.png")
histG2 = cv2.calcHist([altered_green_png], [0], None, [256], [0,256])
# plt.plot(histG2)
# plt.show()

#4e shift green image1 left two pixels
shift_matrix = np.float32([
    [1,0,-2],
    [0,1,0]
])
shifted_green1 = cv2.warpAffine(img1_green, shift_matrix, (img1_green.shape[1], img1_green.shape[0]))
cv2.imwrite("./output/ps0-4-e-1.png", shifted_green1)

#4f subtracted image
subtract_image = img1_green - shifted_green1
cv2.imwrite("./output/ps0-4-f-1.png", subtract_image)

#5a guassian noise
noise = np.random.rand(163,512)*25
noisy_green = green_channel + noise
noisy_img1 = cv2.imread("./output/ps0-1-a-1.png")
noisy_img1[:,:,1] = noisy_green
cv2.imwrite("./output/ps0-5-a-1.png", noisy_img1)

#5b histogram of noise
histG3 = cv2.calcHist([noisy_img1], [0], None, [256], [0,256])
# plt.plot(histG3)
# plt.show()

#5c noise in blue channel
noisy_blue = blue_channel + noise
noisy_img1_blue = cv2.imread("./output/ps0-1-a-1.png")
noisy_img1_blue[:,:,1] = noisy_blue
cv2.imwrite("./output/ps0-5-c-1.png", noisy_img1_blue)

#5e
filtered = cv2.medianBlur(noisy_img1, 5)
filtered_green = filtered[:,:,1]
cv2.imwrite("./output/ps0-5-e-1.png", filtered)

histG4 = cv2.calcHist([filtered_green], [0], None, [256], [0,256])
# plt.plot(histG4)
# plt.show()

#5f guassian blur
filteredG = cv2.GaussianBlur(noisy_img1, (5,5), 25)
cv2.imwrite("./output/ps0-5-f-1.png", filteredG)
filtered_green_G = filteredG[:,:,1]
histG5 = cv2.calcHist([filtered_green_G], [0], None, [256], [0,256])
plt.plot(histG5)
plt.show()