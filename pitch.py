import numpy as np
from itertools import product
from dataclasses import dataclass, field

@dataclass
class Pitch:
    def get_intersections(self, scale=True):
        intersection_points = list(product(self.vert_lines.keys(), self.horiz_lines.keys()))
        scaler = self.scaler(scale)
        intersections = {'_'.join([vl, hl]): (self.vert_lines[vl]*scaler, self.horiz_lines[hl]*scaler)
                        for vl, hl in intersection_points}
        return intersections

    def get_lines(self):
        return list(sorted(set(self.vert_lines).union(set(self.horiz_lines))))

    def scaler(self, convert):
        return self.SCALE if convert else 1

@dataclass
class FootballPitch(Pitch):
    SCALE: int = 5
    X_SIZE: float = 105
    Y_SIZE: float = 68
    GOAL: float = field(default=7.32, init=False)
    BOX_HEIGHT: float = field(default=16.5*2+7.32, init=False)
    BOX_WIDTH: float = field(default=16.5, init=False)
    GOAL_AREA_WIDTH: float = field(default=5.5, init=False)
    GOAL_AREA_HEIGHT: float = field(default=5.5*2+7.32, init=False)

    def __post_init__(self):
        self.vert_lines = {'LG': 0,
                           'LGA': self.GOAL_AREA_WIDTH,
                           'LPA': self.BOX_WIDTH,
<<<<<<< HEAD:BirdsPyView/pitch.py
                           'M': self.X_SIZE/2,
=======
                           'LCC': self.X_SIZE/2-9.15,
                           'M': self.X_SIZE/2,
                           'RCC': self.X_SIZE/2+9.15,
>>>>>>> streamlit_test:pitch.py
                           'RPA': self.X_SIZE-self.BOX_WIDTH,
                           'RGA': self.X_SIZE-self.GOAL_AREA_WIDTH,
                           'RG': self.X_SIZE
                           }

        self.horiz_lines =  {'DG': self.BOX_HEIGHT,
                             'UP': (self.Y_SIZE-self.BOX_HEIGHT)/2,
                             'UG': (self.Y_SIZE-self.GOAL_AREA_HEIGHT)/2,
                             'DG': (self.Y_SIZE+self.GOAL_AREA_HEIGHT)/2,
                             'DP': (self.Y_SIZE+self.BOX_HEIGHT)/2,
                             'U': 0,
                             'D': self.Y_SIZE,
                             'C': self.Y_SIZE/2
                             }


    def get_penalty_area(self, convert=True):
        SPACE = (self.Y_SIZE-self.BOX_HEIGHT)/2
        PENALTY_AREA = [[self.BOX_WIDTH, SPACE],
                        [self.BOX_WIDTH, self.BOX_HEIGHT+SPACE],
                        [0, SPACE],
                        [0, self.BOX_HEIGHT+SPACE]
                       ]
        scaler = self.SCALE if convert else 1
        return np.array(PENALTY_AREA)*scaler


@dataclass
class BasketballPitch(Pitch):
    SCALE: int = 18
    X_SIZE: float = 28.7
    Y_SIZE: float = 15.2