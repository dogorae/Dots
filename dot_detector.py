import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv

img = cv.imread('.\resources\dots.png')
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
gray_blurred = cv.blur(gray, (4, 4))
  
# Apply Hough transform on the blurred image.
detected_circles = cv.HoughCircles(gray_blurred, 
                   cv.HOUGH_GRADIENT, 1, 12, param1 = 5,
                                   param2 = 5, minRadius = 0, maxRadius = 5)
  
# Draw circles that are detected.
if detected_circles is not None:
    detected_circles = np.uint16(np.around(detected_circles))
    
    for pt in detected_circles[0, :]:
        a, b, r = pt[0], pt[1], pt[2]
  
        # Draw the circumference of the circle.
        cv.circle(img, (a, b), r, (0, 255, 0), 2)
  
        # Draw a small circle (of radius 1) to show the center.
        cv.circle(img, (a, b), 1, (0, 0, 255), 3)

plt.imshow(img)
plt.show()
