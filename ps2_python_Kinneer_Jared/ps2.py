import numpy as np
import cv2
import hough_lines_acc as hough
import hough_peaks as pks
import hough_lines_draw as draw
import hough_circles_acc as circ

img = cv2.imread("./input/ps2-input0.png", cv2.IMREAD_GRAYSCALE)

edges = cv2.Canny(img, np.min(img), np.max(img))

#2a
# h, theta, rho = hough.hough_lines_acc(edges)
# cv2.imwrite("./output/ps2-2-a-1.png", h_circled)
# cv2.imshow("hough peaks" ,h)
# cv2.waitKey()
# cv2.destroyAllWindows()

#2b
# peaks = pks.hough_peaks(h, 10)
# h_circled = h
# for peak in peaks:
#     h_circled = cv2.circle(h_circled, (peak[1], peak[0]), 20, 255, 2)
# cv2.imwrite("./output/ps2-2-b-1.png", h_circled)
# cv2.imshow("hough peaks circle" ,h_circled)
# cv2.waitKey()
# cv2.destroyAllWindows()

#2c
# lines = cv2.imread("./input/ps2-input0.png")
# draw.hough_lines_draw(lines, peaks, rho, theta)

#3ab
noisy = cv2.imread("./input/ps2-input0-noise.png", cv2.IMREAD_GRAYSCALE)
blurred = cv2.GaussianBlur(noisy, (3, 3), 0)

noisy_Edges = cv2.Canny(noisy, np.min(noisy), np.max(noisy))
blurred_edges = cv2.Canny(blurred, np.min(blurred), np.max(blurred))

# cv2.imshow("edged", blurred_edges)
# cv2.imwrite("./output/ps2-3-b-1.png", noisy_Edges)
# cv2.imwrite("./output/ps2-3-b-2.png", blurred_edges)

# cv2.waitKey()
# cv2.destroyAllWindows()

#3c
# h, theta, rho = hough.hough_lines_acc(blurred_edges)
# peaks = pks.hough_peaks(h, 10)
# h_circled = h
# for peak in peaks:
#     h_circled = cv2.circle(h_circled, (peak[1], peak[0]), 20, 255, 2)
# cv2.imwrite("./output/ps2-3-c-1.png", h_circled)
# lines = cv2.imread("./input/ps2-input0-noise.png")
# draw.hough_lines_draw(lines, peaks, rho, theta, location='ps2-3-c-2.png')

#4
coins = cv2.imread("./input/ps2-input1.png", cv2.IMREAD_GRAYSCALE)
coins_smoothed = cv2.GaussianBlur(coins, (7,7), 0)
# cv2.imwrite("./output/ps2-4-a-1.png", coins_smoothed)

coins_edges = cv2.Canny(coins_smoothed, np.min(coins_smoothed), np.max(coins_smoothed))
# cv2.imwrite("./output/ps2-4-b-1.png", coins_edges)

# h, theta, rho = hough.hough_lines_acc(coins_edges)
# peaks = pks.hough_peaks(h, 10)
# h_circled = h
# for peak in peaks:
#     h_circled = cv2.circle(h_circled, (peak[1], peak[0]), 20, 255, 2)
# cv2.imwrite("./output/ps2-4-c-1.png", h_circled)
# lines = cv2.imread("./input/ps2-input1.png")
# draw.hough_lines_draw(lines, peaks, rho, theta, location='ps2-4-c-2.png')

#5
cv2.imwrite("./output/ps2-5-a-1.png", coins_smoothed)
h = circ.hough_circles_acc(coins_edges, 20)
peaks = pks.hough_peaks(h, 10)
h_circled = h[0:2]
for peak in peaks:
    h_circled = cv2.circle(h_circled, (peak[1], peak[0]), 30, 255, 1)
    
cv2.imshow("testing", h_circled)
cv2.waitKey(0)
cv2.destroyAllWindows()