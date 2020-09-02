import cv2
import numpy as np

def calculate_homography(pts_src, pts_dst):
    pts_src = np.array(pts_src)
    pts_dst = np.array(pts_dst)
    return cv2.findHomography(pts_src, pts_dst)

def convert_PIL_to_openCV(image):
    return np.array(image)

def apply_homography_to_image(h, img):
    im_out = cv2.warpPerspective(convert_PIL_to_openCV(img), h, (525, 340))
    return im_out

def line_intersect(si1, si2):
    m1, b1 = si1
    m2, b2 = si2
    if m1 == m2:
        return None
    x = (b2 - b1) / (m1 - m2)
    y = m1 * x + b1
    return x/0.8,y/0.8

def get_si_from_coords(lines):
    x1, y1, x2, y2 = lines.T
    slope = (y2-y1) / (x2-x1)
    intercept = y2-slope*x2
    return slope, intercept
