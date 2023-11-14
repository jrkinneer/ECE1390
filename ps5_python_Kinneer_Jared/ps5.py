import numpy as np
import cv2
import random

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
        cv2.circle(output_image, (pair[1], pair[0]), 5, (0,255,0), 2)
  
THRESHOLD = .3
WINDOW_SIZE = 5
suppressed_transA = non_maximal_suppression(harris_transA, WINDOW_SIZE, THRESHOLD)
corners_transA = highlight_corners(suppressed_transA, WINDOW_SIZE, THRESHOLD)
highlighted_transA = cv2.imread("./input/transA.jpg")
draw_corners(corners_transA, highlighted_transA)
cv2.imwrite("./output/ps5-1-c-1.png", highlighted_transA)

suppressed_transB = non_maximal_suppression(harris_transB, WINDOW_SIZE, THRESHOLD)
corners_transB = highlight_corners(suppressed_transB, WINDOW_SIZE, THRESHOLD)
highlighted_transB = cv2.imread("./input/transB.jpg")
draw_corners(corners_transB, highlighted_transB)
cv2.imwrite("./output/ps5-1-c-2.png", highlighted_transB)

suppressed_simA = non_maximal_suppression(harris_simA, WINDOW_SIZE, THRESHOLD)
corners_simA = highlight_corners(suppressed_simA, WINDOW_SIZE, THRESHOLD)
highlighted_simA = cv2.imread("./input/simA.jpg")
draw_corners(corners_simA, highlighted_simA)
cv2.imwrite("./output/ps5-1-c-3.png", highlighted_simA)

suppressed_simB = non_maximal_suppression(harris_simB, WINDOW_SIZE, THRESHOLD)
corners_simB = highlight_corners(suppressed_simB, WINDOW_SIZE, THRESHOLD)
highlighted_simB = cv2.imread("./input/simB.jpg")
draw_corners(corners_simB, highlighted_simB)
cv2.imwrite("./output/ps5-1-c-4.png", highlighted_simB)

#2a
def compute_angle(grad_x, grad_y):
    magnitude, angle = cv2.cartToPolar(grad_y, grad_x, angleInDegrees=True)
    return (magnitude,angle)

def show_points_angles(corners, angles, original_img):
    key_points = []
    
    for i in range(len(corners)):
        x,y = corners[i]
        
        angle = angles[x,y]
            
        kp = cv2.KeyPoint(x=corners[i][1], y=corners[i][0], size=20, angle=angle, response=0, octave=0, class_id=0)
        key_points.append(kp)
        
    return cv2.drawKeypoints(original_img, key_points, outImage=None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

magnitude_transA, transA_angles = compute_angle(grad_y_transA, grad_x_transA)
angled_transA = show_points_angles(corners_transA, transA_angles, transA.copy())
magnitude_transB, transB_angles = compute_angle(grad_y_transB, grad_x_transB)
angled_transB = show_points_angles(corners_transB, transB_angles, transB.copy())
cv2.imwrite("./output/ps5-2-a-1.png", np.hstack((angled_transA, angled_transB)))
magnitude_simA, simA_angles = compute_angle(grad_y_simA, grad_x_simA)
angled_simA = show_points_angles(corners_simA, simA_angles, simA.copy())
magnitude_simA, simB_angles = compute_angle(grad_y_simB, grad_x_simB)
angled_simB = show_points_angles(corners_simB, simB_angles, simB.copy())
cv2.imwrite("./output/ps5-2-a-2.png", np.hstack((angled_simA, angled_simB)))

#2b
def key_points(corners):
    points = []
    for i in range(len(corners)):
        x,y = corners[i]
        kp = cv2.KeyPoint(x=x, y=y, size=3, angle=90, octave = 0)
        points.append(kp)
        
    return points
    
def points_Descriptors(I, points):
    sift = cv2.SIFT_create()
    points_final, descriptors = sift.compute(I, points)
    return points_final, descriptors

def feature_matching(d_a, d_b):
    bfm = cv2.BFMatcher()
    matches = bfm.match(d_a, d_b)
    
    return matches

key_points1 = key_points(corners_transA)
key_points2 = key_points(corners_transB)

pointsA, descA = points_Descriptors(transA, key_points1)
pointsB, descB = points_Descriptors(transB, key_points2)
matches = feature_matching(descA, descB)

def highlight_matches(imgA, imgB, pointsA, pointsB, matches):
    
    width_img1 = int(imgA.shape[1])
    combined_img = np.hstack((imgA, imgB))
    for point in matches:
        p1 = pointsA[point.queryIdx].pt
        p2 = pointsB[point.trainIdx].pt
        
        p1 = (int(p1[0]), int(p1[1]))
        p2 = (int(p2[0]), int(p2[1]))
        
        cv2.circle(combined_img, (p1[1], p1[0]), 5, (255,0,0), 2)
        cv2.circle(combined_img, (p2[1] + width_img1, p2[0]), 5, (255,0,0), 2)
        cv2.line(combined_img, (p1[1], p1[0]), (p2[1] + width_img1, p2[0]), (255,0,0), 2)
        
    return combined_img

A = cv2.imread("./input/transA.jpg")
B = cv2.imread("./input/transB.jpg")
lined = highlight_matches(A, B, pointsA, pointsB, matches)   
cv2.imwrite("./output/ps5-2-b-1.png", lined)


key_points1 = key_points(corners_simA)
key_points2 = key_points(corners_simB)

pointsA_s, descA = points_Descriptors(simA, key_points1)
pointsB_s, descB = points_Descriptors(simB, key_points2)
matches_s = feature_matching(descA, descB)
A_s = cv2.imread("./input/simA.jpg")
B_s = cv2.imread("./input/simB.jpg")
lined = highlight_matches(A_s, B_s, pointsA_s, pointsB_s, matches_s)   
cv2.imwrite("./output/ps5-2-b-2.png", lined)

#3
def translational_case(imgA, imgB, pointsA, pointsB, matches):
    x = random.randint(0, len(matches))
    p1_first = pointsA[matches[x].queryIdx].pt
    p2_first = pointsB[matches[x].trainIdx].pt
    
    trans_distance_first = np.sqrt((p1_first[0]-p2_first[0])**2 - (p1_first[1] - p2_first[1]**2))
    
    consensus = []
    
    final_dist = trans_distance_first
    #get first consensus
    for point in matches:
        p1 = pointsA[point.queryIdx].pt
        p2 = pointsB[point.trainIdx].pt
        
        dist = np.sqrt((p1[0]-p2[0])**2 - (p1[1] - p2[1]**2))
        
        if (dist < trans_distance_first * 1.2 and dist > trans_distance_first * .8):
            consensus.append(point)
        
    #see if any other pair distance creates a better consensus
    for point in matches:
        p1_x = pointsA[point.queryIdx].pt
        p2_x = pointsB[point.trainIdx].pt
    
        trans_distance_x = np.sqrt((p1_x[0]-p2_x[0])**2 - (p1_x[1] - p2_x[1]**2))
        
        temp_consensus = []
        
        for point2 in matches:
            p1_y = pointsA[point2.queryIdx].pt
            p2_y = pointsB[point2.trainIdx].pt
        
            dist = np.sqrt((p1_y[0]-p2_y[0])**2 - (p1_y[1] - p2_y[1]**2))
            
            if (dist < trans_distance_x * 1.2 and dist > trans_distance_x * .8):
                temp_consensus.append(point2) 
    
        if (len(temp_consensus) > len(consensus)):
            consensus = temp_consensus
            final_dist = trans_distance_x
            
    final_img = np.hstack((imgA, imgB))
    width_img1 = imgA.shape[1]
    
    for point in consensus:
        p1 = pointsA[point.queryIdx].pt
        p2 = pointsB[point.trainIdx].pt
        
        p1 = (int(p1[0]), int(p1[1]))
        p2 = (int(p2[0]), int(p2[1]))
        
        cv2.circle(final_img, (p1[1], p1[0]), 5, (255,0,0), 2)
        cv2.circle(final_img, (p2[1] + width_img1, p2[0]), 5, (255,0,0), 2)
        cv2.line(final_img, (p1[1], p1[0]), (p2[1] + width_img1, p2[0]), (255,0,0), 2)
    
    print("translational vector length = ", final_dist)
    print("largest consensus % = ", len(consensus)/len(matches))
    
    return final_img

cv2.imwrite("./output/ps5-3-a-1.png", translational_case(A, B, pointsA, pointsB, matches))

# def sim_case(imgA, imgB, pointsA, pointsB, mathces):