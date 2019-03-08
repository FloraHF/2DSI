from Config import Config
from ProcessPlayer import ProcessPlayer
from Experience import Experience
from NetworkHH import NetworkHH

from multiprocessing import Value


class ProcessIntruder(ProcessPlayer):
    """docstring for ProcessIntruder."""
    def __init__(self, env, id, state, episode_count):
        super(ProcessIntruder, self).__init__(env, id, state, episode_count)

        self.role = 'intruder'

        self.action_space = Config.INTRUDER_ACTION_SPACE
        self.nb_actions = self.get_num_actions()
        self.vmax = Config.INTRUDER_MAX_VELOCITY

        self.model = NetworkHH(Config.DEVICE, self.role+str(self.id), self.action_space)

        self.captured = Value('i', 0)
        self.entered = Value('i', 0)

    def clearup_reward(self):
        reward = 0
        if self.entered.value:
            reward +=  Config.REWARD_ENTER
        reward -= self.env.world.target.contour(self.state.x, self.state.y)/self.env.world.max_target_level

        if self.captured.value or self.entered.value:
            self.state.done = 1

        return reward

    def step(self, action):

        # settle up reward of the former action
        reward = self.clearup_reward()

        # if not done yet, make move
        if not self.state.done:
            new_x, new_y = self.try_step(action)
            num_trial = 0
            # move, but can't get outside the world map
            while not self.env.world.is_in_world(new_x, new_y) and \
                    num_trial < 2 * self.nb_actions:
                new_x, new_y = self.try_step(self.random_move())
                num_trial += 1
            if num_trial < 2*self.nb_actions:
                self.state.x = new_x
                self.state.y = new_y

            # check capture
            for d in range(len(self.env.defenders)):
                if self.env.defenders[d].is_captured(self.env.defenders[d], self):
                    self.captured.value = 1
                    self.env.defenders[d].capture_buffer.value += 1

            # check enter
            if self.env.world.target.is_in_target(self.state.x, self.state.y):
                self.entered.value = 1
                # every defender gets penalized
                for d in range(len(self.env.defenders)):
                    self.env.defenders[d].enter_buffer.value += 1

        return reward
