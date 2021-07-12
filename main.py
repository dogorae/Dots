import slm_control.slm as slm
import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
import os

# Coordinates in x,y, not row, column
SEED_POS = [[96, 212], [96, 10]]   # pixel coordinates of seed laser on camera (two orders)
SLM_POS = [[635, 615], [1290, 615]] # corresponding SLM pixel numbers
IMG_PATH = 'resources\dots2.png'

def get_rot_and_trans_matrix(src, dst):
    """
    src: list of two points before transformation
    dst: list of two points after transformation
    """
    if len(src) != 2 or len(dst) !=2:
        raise Exception("Need two pairs of points for transformation")
    pt1x_before, pt1y_before = src[0]
    pt2x_before, pt2y_before = src[1]
    pt1x_after, pt1y_after = dst[0]
    pt2x_after, pt2y_after = dst[1]
    r_before = (pt2x_before-pt1x_before) + 1j*(pt2y_before-pt1y_before)
    r_after = (pt2x_after-pt1x_after) + 1j*(pt2y_after-pt1y_after)    
    r_change = r_after / r_before
    angle = np.angle(r_change)
    scale = np.abs(r_change)
    x_trans = pt1x_after \
        - (scale*np.cos(angle)*pt1x_before - scale*np.sin(angle)*pt1y_before)
    y_trans = pt1y_after \
        - (scale*np.sin(angle)*pt1x_before + scale*np.cos(angle)*pt1y_before)
    M = [[scale*np.cos(angle), -scale*np.sin(angle), x_trans],
         [scale*np.sin(angle), scale*np.cos(angle), y_trans]]
    return np.array(M)

def transform_dots(dots, M):
    transformed_dots = list(map(lambda x: np.matmul(M, np.append(x, 1.0)).astype(np.uint32),
                                dots)) # Affine transformation
    return transformed_dots

def circle_finder(img):
    marked_image = img
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    gray_blurred = cv.blur(gray, (4, 4))
    detected_circles = cv.HoughCircles(gray_blurred, 
                                       cv.HOUGH_GRADIENT, 1, 13, param1=20,
                                       param2=3, minRadius=0, maxRadius=5)
    circle_cdts = []
  
    # Draw circles that are detected.
    if detected_circles is not None:
        detected_circles = np.uint16(np.around(detected_circles))
        
        for pt in detected_circles[0, :]:
            a, b, r = pt[0], pt[1], pt[2]
            cv.circle(marked_image, (a, b), r, (0, 255, 0), 2)  # draw circumference
            cv.circle(marked_image, (a, b), 1, (0, 0, 255), 3)  # draw center
            circle_cdts.append(np.array([a,b], dtype=float))

    return circle_cdts, marked_image

dir = os.path.dirname(__file__)
filename = os.path.join(IMG_PATH)
img = cv.imread(filename)
dot_cdts, marked_image = circle_finder(img)
rot_matrix = get_rot_and_trans_matrix(SEED_POS, SLM_POS)
new_dots = transform_dots(dot_cdts, rot_matrix)
canvas = slm.Canvas(1920, 1200)  # represents SLM screen

for dot in new_dots:
    superpixel = slm.Superpixel(pos=dot, width=15, height=15)
    canvas.add_superpixel(superpixel)

filepath = r"C:\santec\SLM-200\Files\grating\test.csv"
canvas.save(filepath)
slm.display(filepath)

fig, (ax1, ax2) = plt.subplots(2, figsize=(5,10))
ax1.imshow(marked_image)
ax1.set_title("Camera image")
ax2.imshow(canvas.canvas, cmap='gray')
ax2.set_title("SLM screen")
fig.tight_layout()
plt.show()
