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

def apply_homography_to_points(h, points, inverse=False):
    if inverse:
        h = np.linalg.inv(h)
    _points = np.hstack([points, np.ones((len(points), 1))])
    _converted_points = np.dot(h,_points.T)
    points = _converted_points/_converted_points[2]
    return points[:2].T

def line_intersect(si1, si2):
    m1, b1 = si1
    m2, b2 = si2
    if m1 == m2:
        return None
    x = (b2 - b1) / (m1 - m2)
    y = m1 * x + b1
    return x,y

def get_si_from_coords(lines):
    x1, y1, x2, y2 = lines.T
    slope = (y2-y1) / (x2-x1)
    intercept = y2-slope*x2
    return slope, intercept
