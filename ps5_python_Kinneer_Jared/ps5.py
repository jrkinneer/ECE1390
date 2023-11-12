import numpy as np
import cv2
import scipy

#1a
def compute_grad(img):
    kernel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    kernel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])

    # Apply kernels using convolution
    gradient_x = np.zeros_like(img, dtype=np.float32)
    gradient_y = np.zeros_like(img, dtype=np.float32)

    # Apply convolution with the Sobel kernels
    for i in range(1, img.shape[0] - 1):
        for j in range(1, img.shape[1] - 1):
            gradient_x[i, j] = np.sum(img[i - 1:i + 2, j - 1:j + 2] * kernel_x)
            gradient_y[i, j] = np.sum(img[i - 1:i + 2, j - 1:j + 2] * kernel_y)

    gradient_x = np.abs(gradient_x)
    gradient_y = np.abs(gradient_y)
    gradient_x = ((gradient_x - 128)/128)*127
    
    gradient_y = ((gradient_y - 128)/128)*127
    
    return (gradient_x, gradient_y)

transA = cv2.imread("./input/transA.jpg",cv2.IMREAD_GRAYSCALE)
simA = cv2.imread("./input/simA.jpg",cv2.IMREAD_GRAYSCALE)

grad_x_transA , grad_y_transA = compute_grad(transA)

final_trans = np.hstack((grad_x_transA, grad_y_transA))

grad_x_simA , grad_y_simA = compute_grad(simA)
final_sim = np.hstack((grad_x_simA, grad_y_simA))
cv2.imwrite("./output/ps5-1-a-1.png", final_trans)
cv2.imwrite("./output/ps5-1-a-2.png", final_sim)

#1b
def harris_value(grad_x, grad_y, window_size, alpha):
    Ix2 = grad_x**2
    Ixy = grad_x*grad_y
    Iy2 = grad_y**2
    
    Sx2 = cv2.blur(Ix2, (window_size, window_size))
    Sxy = cv2.blur(Ixy, (window_size, window_size))
    Sy2 = cv2.blur(Iy2, (window_size, window_size))
    
    determinant = (Sx2 * Sy2) - (Sxy**2)
    trace = Sx2 + Sy2
    harris = determinant - (alpha * (trace**2))
    harris = ((harris - harris.min()) * (1/(harris.max() - harris.min()) * 255)).astype('uint8')
    return harris

def non_maximal_suppression(image, window_size = 3, threshold_percent = .01):
    threshold = threshold_percent*np.max(image)
    suppressed = image.copy()
    z = window_size//2
    count_suppressed = 0
    for i in range(z, image.shape[0] - z):
        for j in range(z, image.shape[1] - z):
            window = suppressed[i - z:i + z + 1, j - z: j + z + 1]
            
            
            if suppressed[i, j] < np.max(window) or suppressed[i,j] < threshold:
                suppressed[i,j] = 0
                count_suppressed += 1
                
    return suppressed

harris_transA = harris_value(grad_x_transA, grad_y_transA, 5, .04)
harris_simA = harris_value(grad_x_simA, grad_y_simA, 5, .04)

#gradients and harris values of transB and simB
transB = cv2.imread("./input/transB.jpg",cv2.IMREAD_GRAYSCALE)
simB = cv2.imread("./input/simB.jpg",cv2.IMREAD_GRAYSCALE)

grad_x_transB , grad_y_transB = compute_grad(transB)
grad_x_simB , grad_y_simB = compute_grad(simB)

harris_transB = harris_value(grad_x_transB, grad_y_transB, 5, .04)
harris_simB = harris_value(grad_x_simB, grad_y_simB, 5, .04)
cv2.imwrite("./output/ps5-1-b-1.png", harris_transA)
cv2.imwrite("./output/ps5-1-b-3.png", harris_simA)
cv2.imwrite("./output/ps5-1-b-2.png", harris_transB)
cv2.imwrite("./output/ps5-1-b-4.png", harris_simB)

#1c
def highlight_corners(harris_image, window_size=3, threshold_percent = .01):
    threshold = np.max(harris_image) * threshold_percent
    harris = harris_image.copy()
    
    z = window_size//2
    
    corners = []
    
    for i in range(z, harris_image.shape[0] - z):
        for j in range(z, harris_image.shape[1] - z):
            
            window = harris[i - z:i + z + 1, j - z: j + z + 1]
            
            if (harris[i, j] > threshold and harris[i,j] == np.max(window)):
                corners.append((i,j))
    
    return corners

def draw_corners(corners, output_image):
    for pair in corners:
        cv2.circle(output_image, (pair[1], pair[0]), 3, (0,255,0), 2)
  
suppressed_transA = non_maximal_suppression(harris_transA, 5, .1)
corners_transA = highlight_corners(suppressed_transA, 5, .3)
highlighted_transA = cv2.imread("./input/transA.jpg")
draw_corners(corners_transA, highlighted_transA)
cv2.imwrite("./output/ps5-1-c-1.png", highlighted_transA)

suppressed_transB = non_maximal_suppression(harris_transB, 5, .1)
corners_transB = highlight_corners(suppressed_transB, 5, .3)
highlighted_transB = cv2.imread("./input/transB.jpg")
draw_corners(corners_transB, highlighted_transB)
cv2.imwrite("./output/ps5-1-c-2.png", highlighted_transB)

suppressed_simA = non_maximal_suppression(harris_simA, 5, .1)
corners_simA = highlight_corners(suppressed_simA, 5, .3)
highlighted_simA = cv2.imread("./input/simA.jpg")
draw_corners(corners_simA, highlighted_simA)
cv2.imwrite("./output/ps5-1-c-3.png", highlighted_simA)

suppressed_simB = non_maximal_suppression(harris_simB, 5, .1)
corners_simB = highlight_corners(suppressed_simB, 5, .3)
highlighted_simB = cv2.imread("./input/simB.jpg")
draw_corners(corners_simB, highlighted_simB)
cv2.imwrite("./output/ps5-1-c-4.png", highlighted_simB)

#2a
def compute_angle(grad_x, grad_y):
    _, angle = cv2.cartToPolar(grad_y, grad_x, angleInDegrees=True)
    return angle

def show_points_angles(corners, angles, original_img):
    
    key_points = []
    
    for i in range(len(corners)):
        angle = angles[0][i]
        
        kp = cv2.KeyPoint(x=corners[i][1], y=corners[i][0], size=20, angle=angle, response=0, octave=0, class_id=0)
        key_points.append(kp)
        
    return cv2.drawKeypoints(original_img, key_points, outImage=None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

transA_angles = compute_angle(grad_y_transA, grad_x_transA)
angled = show_points_angles(corners_transA, transA_angles, transA.copy())
# cv2.imshow("transA with angles", angled)
# cv2.waitKey(0)
# cv2.destroyAllWindows()