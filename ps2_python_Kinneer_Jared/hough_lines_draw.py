import numpy as np
import cv2

def hough_lines_draw(img, peaks, rho, theta, location='ps2-2-c-1.png'):
    temp = img
    for i in range(len(peaks)):
        r = rho[peaks[i][0]]
        t = theta[peaks[i][1]]
        a = np.cos(t)
        b = np.sin(t)
        x0 = a * r
        y0 = b * r
        x1 = int(x0 + 1000*(-b))
        y1 = int(y0 + 1000*(a))
        x2 = int(x0 - 1000*(-b))
        y2 = int(y0 - 1000*(a))
        temp = cv2.line(temp, (x1,y1), (x2,y2), (255,0,0), 5)
        
    cv2.imwrite("./output/" + location, temp)
        
        
    
    return None