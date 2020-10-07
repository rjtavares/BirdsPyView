import cv2
import numpy as np
from shapely.geometry import Polygon
from itertools import product
from PIL import Image, ImageDraw

class Homography():
    def __init__(self, pts_src, pts_dst):
        self.pts_src = np.array(pts_src)
        self.pts_dst = np.array(pts_dst)
        self.h, out = cv2.findHomography(self.pts_src, self.pts_dst)
        self.im_size = (525, 340)
        self.im_width = self.im_size[0]
        self.im_heigth = self.im_size[1]

    def apply_to_image(self, image):
        im_out = cv2.warpPerspective(np.array(image.im), self.h, self.im_size)
        return im_out

    def apply_to_points(self, points, inverse=False, normalize=False):
        h = np.linalg.inv(self.h) if inverse else self.h
        _points = np.hstack([points, np.ones((len(points), 1))])
        _converted_points = np.dot(h,_points.T)
        points = _converted_points/_converted_points[2]
        return points[:2].T

class VoronoiPitch():
    def __init__(self, df):
        self.vor, self.df = calculate_voronoi(df)

    def get_regions(self):
        return [index for index, region in enumerate(self.vor.regions) if (not -1 in region) and (len(region)>0)]
    
    def get_points_region(self, region):
        return np.vstack([self.vor.vertices[i] for i in self.vor.regions[region]])
    
    def get_color_region(self, region):
        return self.df[self.df['region']==region]['team'].values[0]

class PitchImage():
    def __init__(self, image_to_open, pitch, width=600):
        self.im = self.open(image_to_open, width=width)
        self.pitch = pitch

    def open(self, image_to_open, width):
        im = Image.open(image_to_open)
        im = im.resize((width, int(width*im.height/im.width)))
        return im
    
    def set_info(self, df, lines):
        df['line'] = lines
        df['y1_line'] = df['top']+df['y1']
        df['y2_line'] = df['top']+df['y2']
        df['x1_line'] = df['left']+df['x1']
        df['x2_line'] = df['left']+df['x2']
        df['slope'], df['intercept'] = get_si_from_coords(df[['x1_line', 'y1_line', 'x2_line', 'y2_line']].values)
        df = df.set_index('line')
        self.df = df
        self.lines = lines
        self.h = Homography(*self.get_intersections())
        self.conv_im = Image.fromarray(self.h.apply_to_image(self))
        self.coord_converter = np.array(self.h.im_size)/100

    def get_intersections(self):
        lines = self.lines
        vertical_lines = [x for x in lines if x in self.pitch.vert_lines]
        horizontal_lines = [x for x in lines if x in self.pitch.horiz_lines]
        intersections = {'_'.join([v, h]): line_intersect(self.df.loc[v, ['slope', 'intercept']], self.df.loc[h, ['slope', 'intercept']])
                            for v,h in product(vertical_lines, horizontal_lines)}

        pts_src = list(intersections.values())
        pts_dst = [self.pitch.get_intersections()[x] for x in intersections]

        return pts_src, pts_dst

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

def get_polygon(points, image, convert):
    base_polygon = Polygon(points.tolist())
    pitch_polygon = Polygon(((0,0), (0,image.h.im_heigth), (image.h.im_width,image.h.im_heigth), (image.h.im_width,0)))
    camera_polygon = Polygon(image.h.apply_to_points(((0,0), (0,image.im.height), (image.im.width, image.im.height), (image.im.width,0))))
    polygon = camera_polygon.intersection(pitch_polygon).intersection(base_polygon)
    if convert:
        polygon = image.h.apply_to_points(np.vstack(polygon.exterior.xy).T, inverse=True)
    else:
        polygon = np.vstack(polygon.exterior.xy).T
    return polygon