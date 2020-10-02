from helpers import Homography
import numpy as np

def test_homography():
    pts_src = [[160.0444,  34.4228],
                [408.3040, 199.1448],
                [342.0459,  18.1934],
                [670.2333, 163.3532],
                ]

    pts_dst = [[442.5000,  69.2000],
                [442.5000, 270.8000],
                [525,       69.2000],
                [525,      270.8000],
                ]
    h = Homography(pts_src, pts_dst)
    valid_h = np.array([[4.67359060e-01,  1.03886936e+00,  3.95244335e+02],
                        [1.77165177e-01,  2.19878042e+00, -2.49428342e+01],
                        [6.88618532e-05,  3.83575476e-03,  1.00000000e+0]])

    assert np.allclose(h.h, valid_h)
    assert np.allclose(h.apply_to_points([[0,0], [10,50]]),
                       np.array([[395.2443349,  -24.94283416], [378.92692043,  72.76273286]]))