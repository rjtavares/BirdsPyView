import numpy as np
from dataclasses import dataclass, field

@dataclass
class FootballPitch:
    SCALE: int = 5
    X_SIZE: int = 105
    Y_SIZE: int = 68
    GOAL: float = field(default=7.32, init=False)
    BOX_HEIGHT: float = field(default=16.5*2+7.32, init=False)
    BOX_WIDTH: float = field(default=16.5, init=False)

    def get_penalty_area(self):
        SPACE = (self.Y_SIZE-self.BOX_HEIGHT)/2
        PENALTY_AREA = [[self.BOX_WIDTH, SPACE],
                        [self.BOX_WIDTH, self.BOX_HEIGHT+SPACE],
                        [0, SPACE],
                        [0, self.BOX_HEIGHT+SPACE]
                       ]

        return np.array(PENALTY_AREA)*self.SCALE