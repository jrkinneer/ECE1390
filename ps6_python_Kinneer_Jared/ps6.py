import cv2
import numpy as np
from lucas_kanade import lucas_kanade as LK
import matplotlib.pyplot as plt
from reduce import gaussian_reduce
from expand import expand
#1a
#lucas kanade optic flow
origin_image = cv2.imread("./input/TestSeq/Shift0.png", cv2.IMREAD_GRAYSCALE)
second_image = cv2.imread("./input/TestSeq/ShiftR2.png", cv2.IMREAD_GRAYSCALE)

# u,v = LK(origin_image, second_image, 5)

# plt.figure()
# plt.imshow(u**2 + v**2, cmap='jet')
# plt.colorbar()
# plt.title('displacement u**2 + v**2 ShiftR2')
# plt.savefig("./output/ps6-1-a-1.png")
# # plt.show()

# second_image = cv2.imread("./input/TestSeq/ShiftR5U5.png", cv2.IMREAD_GRAYSCALE)
# u,v = LK(origin_image, second_image, 5)
# plt.figure()
# plt.imshow(u**2 + v**2, cmap='jet')
# plt.colorbar()
# plt.title('displacement u**2 + v**2 ShiftR5U5')
# plt.savefig("./output/ps6-1-a-2.png")
# # plt.show()

# #1b
# second_image = cv2.imread("./input/TestSeq/ShiftR10.png", cv2.IMREAD_GRAYSCALE)
# u,v = LK(origin_image, second_image, 5)
# plt.figure()
# plt.imshow(u**2 + v**2, cmap='jet')
# plt.colorbar()
# plt.title('displacement u**2 + v**2 ShiftR10')
# plt.savefig("./output/ps6-1-b-1.png")

# second_image = cv2.imread("./input/TestSeq/ShiftR20.png", cv2.IMREAD_GRAYSCALE)
# u,v = LK(origin_image, second_image, 5)
# plt.figure()
# plt.imshow(u**2 + v**2, cmap='jet')
# plt.colorbar()
# plt.title('displacement u**2 + v**2 ShiftR20')
# plt.savefig("./output/ps6-1-b-2.png")

# second_image = cv2.imread("./input/TestSeq/ShiftR40.png", cv2.IMREAD_GRAYSCALE)
# u,v = LK(origin_image, second_image, 5)
# plt.figure()
# plt.imshow(u**2 + v**2, cmap='jet')
# plt.colorbar()
# plt.title('displacement u**2 + v**2 ShiftR40')
# plt.savefig("./output/ps6-1-b-3.png")

#2
frame1 = cv2.imread("./input/DataSeq1/yos_img_01.jpg", cv2.IMREAD_GRAYSCALE)
first_reduce = gaussian_reduce(frame1)
second_reduce = gaussian_reduce(first_reduce)
third_reduce = gaussian_reduce(second_reduce)

f, ax = plt.subplots(2,2)
ax[0,0].imshow(frame1, cmap='Greys_r')
ax[0,0].set_title("original frame")
ax[0,1].imshow(first_reduce, cmap='Greys_r')
ax[0,1].set_title("first reduction")
ax[1,0].imshow(second_reduce, cmap='Greys_r')
ax[1,0].set_title("second reduction")
ax[1,1].imshow(third_reduce, cmap='Greys_r')
ax[1,1].set_title("third reduction")
f.tight_layout(pad=2)
plt.savefig("./output/ps6-2-a-1.png")
plt.show()

first_exp = expand(third_reduce, second_reduce.shape)
second_expansion = expand(first_exp, first_reduce.shape)
third_expansion = expand(second_expansion, frame1.shape)

first_laplacian = first_exp - second_reduce
second_laplacian = second_expansion - first_reduce
third_laplacian = third_expansion - frame1

f_2, ax_2 = plt.subplots(2, 2)
ax_2[0,0].imshow(third_reduce, cmap='Greys_r')
ax_2[0,0].set_title("final gaussian")
ax_2[0,1].imshow(first_laplacian, cmap='Greys_r')
ax_2[0,1].set_title("first laplacian")
ax_2[1,0].imshow(second_laplacian, cmap='Greys_r')
ax_2[1,0].set_title("second laplacian")
ax_2[1,1].imshow(third_laplacian, cmap='Greys_r')
ax_2[1,1].set_title("third laplacian")
f_2.tight_layout(pad=2)
plt.savefig("./output/ps6-2-b-1.png")
# plt.show()

#3
#image flow
# img1 = cv2.imread("./input/DataSeq1/yos_img_01.png", cv2.IMREAD_GRAYSCALE)
# img2 = cv2.imread("./input/DataSeq1/yos_img_02.png", cv2.IMREAD_GRAYSCALE)
# img3 = cv2.imread("./input/DataSeq1/yos_img_03.png", cv2.IMREAD_GRAYSCALE)

# data_seq = [img1, img2, img3]
# for img in data_seq:
    