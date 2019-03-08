import numpy as np

from Config import Config
from ProcessPlayer import ProcessPlayer
from Experience import Experience
from NetworkHH import NetworkHH

from multiprocessing import Value


class ProcessDefender(ProcessPlayer):

    def __init__(self, env, id, state, episode_count):
        super(ProcessDefender, self).__init__(env, id, state, episode_count)

        self.role = 'defender'
        self.capture_range = Value('d', Config.CAPTURE_RANGE)
        self.max_capture_level = self.get_max_capture_level()

        self.action_space = Config.DEFENDER_ACTION_SPACE
        self.nb_actions = self.get_num_actions()
        self.vmax = Config.DEFENDER_MAX_VELOCITY

        self.model = NetworkHH(Config.DEVICE, self.role+str(self.id), self.action_space)

        self.capture_buffer = Value('i', 0)
        self.enter_buffer = Value('i', 0)

    def capture_level(self, x, y):
        return np.sqrt((x - self.state.x)**2 + (y - self.state.y)**2) - self.capture_range.value

    def get_max_capture_level(self):
        return np.sqrt((2*self.env.world.x_bound)**2 + (2*self.env.world.y_bound )**2) - self.capture_range.value

    def is_captured(self, d, i):
        return not (np.sqrt((d.state.x - i.state.x)**2 + (d.state.y - i.state.y)**2) - self.capture_range.value) > 0

    def clearup_reward(self, intruders):

        reward = 0

        active_count = 0
        for i in intruders:
            if not i.captured.value and not i.entered.value:
                active_count += 1
                reward -= self.capture_level(i.state.x, i.state.y)/self.max_capture_level
                reward += self.env.world.target.contour(i.state.x, i.state.y)/self.env.world.max_target_level
        if active_count:
            reward /= active_count
        else:
            self.state.done = 1

        # terminal: reward for capture, penalty for entering
        reward += Config.REWARD_CAPTURE * self.capture_buffer.value
        reward -= Config.REWARD_ENTER * self.enter_buffer.value
        self.capture_buffer.value = 0
        self.enter_buffer.value = 0

        # return
        return reward

    def step(self, action):

        # settle up reward of the former action
        reward = self.clearup_reward(self.env.intruders)

        # if not done yet, make move
        if not self.state.done:
            # have to take some actions
            self.time_buffer = 1
            # try to take an action, but can't enter the target
            new_x, new_y = self.try_step(action)
            num_trial = 0
            while (self.env.world.target.is_in_target(new_x, new_y) or \
                  (not self.env.world.is_in_world(new_x, new_y))) and \
                    num_trial < 2*self.nb_actions:
                new_x, new_y = self.try_step(self.random_move())
                num_trial += 1
            if num_trial < 2*self.nb_actions:
                self.state.x = new_x
                self.state.y = new_y

            for i in range(len(self.env.intruders)):
                if not self.env.intruders[i].captured.value and not self.env.intruders[i].entered.value:
                    if self.is_captured(self, self.env.intruders[i]):
                        self.env.intruders[i].captured.value = 1
                        self.capture_buffer.value += 1

        # print('defender', id, 'done:', self.defenders[id].done)
        return reward
