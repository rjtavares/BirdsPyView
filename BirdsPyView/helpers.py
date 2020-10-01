import cv2
import numpy as np

class Homography():
    def __init__(self, pts_src, pts_dst):
        self.pts_src = np.array(pts_src)
        self.pts_dst = np.array(pts_dst)
        self.h, out = cv2.findHomography(self.pts_src, self.pts_dst)
        self.im_size = (525, 340)

    def apply_to_image(self, img):
        im_out = cv2.warpPerspective(np.array(img), self.h, self.im_size)
        return im_out

    def apply_to_points(self, points, inverse=False):
        h = np.linalg.inv(self.h) if inverse else self.h
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

def calculate_voronoi(df):
    from scipy.spatial import Voronoi
    values = np.vstack((df[['x', 'y']].values,
                        [-1000,-1000],
                        [+1000,+1000],
                        [+1000,-1000],
                        [-1000,+1000]
                       ))

    vor = Voronoi(values)

    df['region'] = vor.point_region[:-4]

    return vor, df