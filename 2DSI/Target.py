from Config import Config
import numpy as np

class Target:
    """This is a for Target."""
    def __init__(self):
        self.shape = Config.TARGET_TYPE
        if self.shape == 'circle':
            self.r = Config.TARGET_RADIUS
            self.c = Config.TARGET_CENTER

    def contour(self, x, y):
        if self.shape == 'circle':
            level = np.sqrt((x - self.c[0])**2 + (y - self.c[1])**2) - self.r
        return level

    def is_in_target(self, x, y):
        return (self.contour(x, y) < 0)
