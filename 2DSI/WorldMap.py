from Target import Target
from Config import Config

class WorldMap():
    """docstring for WorldMap."""
    def __init__(self):
        self.x_bound = Config.WORLD_X_BOUND
        self.y_bound = Config.WORLD_Y_BOUND
        self.shape = 'Square'
        self.target = Target()
        self.max_target_level = self.get_max_target_level()

    def is_in_world(self, x, y):
        if self.shape == 'Square':
            return abs(x)<self.x_bound and abs(y)<self.y_bound

    def get_max_target_level(self):
        return self.target.contour(self.x_bound, self.y_bound)
