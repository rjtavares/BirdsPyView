import numpy as np

class FootballPitch:
    def __init__(self):
        self.SCALE = 5
        self.X_SIZE = 105
        self.Y_SIZE = 68
        self.GOAL = 7.32
        self.BOX_HEIGHT = 16.5*2 + self.GOAL
        self.BOX_WIDTH = 16.5

    def get_penalty_area(self):
        SPACE = (self.Y_SIZE-self.BOX_HEIGHT)/2
        PENALTY_AREA = [[self.BOX_WIDTH, SPACE],
                        [self.BOX_WIDTH, self.BOX_HEIGHT+SPACE],
                        [0, SPACE],
                        [0, self.BOX_HEIGHT+SPACE]
                       ]

        return np.array(PENALTY_AREA)*self.SCALE