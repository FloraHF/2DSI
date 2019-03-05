from multiprocessing import Process, Queue, Value
from datetime import datetime
import time

import numpy as np
import random as rd
import math

from Config import Config
from Experience import Experience
from ThreadTrainer import ThreadTrainer
from ThreadPredictor import ThreadPredictor

class ProcessPlayer(Process):
    """docstring for ProcessPlayer."""
    def __init__(self, env, id, state, episode_log_q):
        super(ProcessPlayer, self).__init__()

        self.env = env
        self.id = id
        self.state = state

        self.episode_log_q = episode_log_q
        self.prediction_q = Queue(maxsize=Config.MAX_QUEUE_SIZE)
        self.training_q = Queue(maxsize=Config.MAX_QUEUE_SIZE)
        self.wait_q = Queue(maxsize=1)

        self.experiences = []

        self.predictor = ThreadPredictor(self)
        self.trainer = ThreadTrainer(self)

        # learning parameters
        self.discount_factor = Config.DISCOUNT
        self.exit_flag = Value('i', 0)

    def reset(self):
        self.state.done = 0

    def get_num_actions(self):
        return(len(self.action_space))

    def random_move(self):
        return self.action_space[rd.randint(0, self.get_num_actions()-1)]

    def try_step(self, action):
        x = self.state.x + Config.TIME_STEP * self.vmax * math.cos(action)
        y = self.state.y + Config.TIME_STEP * self.vmax * math.sin(action)
        return x, y

###################################################################################
############################ Learning Helper Funcs ################################
###################################################################################
    def predict(self, state):
        # put the state in the prediction q
        self.prediction_q.put(state)
        # wait for the prediction to come back
        action, value = self.wait_q.get()
        return action, value

    def train_model(self, x, a, y, r):
        self.model.train(x, a, y, r)

    @staticmethod
    def _accumulate_rewards(experiences, discount_factor, terminal_reward):
        reward_sum = terminal_reward
        for t in reversed(range(0, len(experiences)-1)):
            r = np.clip(experiences[t].reward, Config.REWARD_MIN, Config.REWARD_MAX)
            reward_sum = discount_factor * reward_sum + r
            experiences[t].reward = reward_sum
        return experiences[:-1]

    def convert_exp(self, experiences):
        x_ = np.array([exp.previous_state for exp in experiences])
        a_ = np.array([exp.action for exp in experiences])
        y_ = np.array([exp.current_state for exp in experiences])
        r_ = np.array([exp.reward for exp in experiences])
        # print(x_, r_, a_)
        return x_, a_, y_, r_

    def convert_state(self, dstate, istate):
        state = np.empty((Config.PLAYER_DIMENSION*(self.env.dcount+self.env.icount)))
        for d in range(self.env.dcount):
            state[Config.PLAYER_DIMENSION*d]   = dstate[d].x
            state[Config.PLAYER_DIMENSION*d+1] = dstate[d].y
            state[Config.PLAYER_DIMENSION*d+2] = dstate[d].done
        for i in range(self.env.icount):
            state[Config.PLAYER_DIMENSION*(self.env.dcount+i)]   = istate[i].x
            state[Config.PLAYER_DIMENSION*(self.env.dcount+i)+1] = istate[i].y
            state[Config.PLAYER_DIMENSION*(self.env.dcount+i)+2] = istate[i].done
        return state

###################################################################################
################################# Run Learning ####################################
###################################################################################
    def run_episode(self):
        reward_sum = 0.0
        experiences = []
        previous_state = self.convert_state(self.env.dstates, self.env.istates)

        moves = 0
        while not self.state.done:
            moves += 1
            action, value = self.predict(previous_state)
            reward = self.step(action)
            current_state = self.convert_state(self.env.dstates, self.env.istates)
            if Config.PLAY_MODE:
                self.trj_log_q.put((self.role, self.id, self.state.x, self.state.y, action, reward))
            reward_sum += reward
            # if moves % 20 == 0 or self.state.done:
            #     print(self.role, self.id, self.state.x, self.state.y, self.state.done)
            # very first step
            if not len(experiences):
                exp = Experience(previous_state, action, current_state, reward)
                experiences.append(exp)
                previous_state = current_state
                continue

            experiences[-1].reward = reward
            exp = Experience(previous_state, action, current_state, reward)
            experiences.append(exp)
            updated_exps = ProcessPlayer._accumulate_rewards(experiences, self.discount_factor, value)
            x_, a_, y_, r_ = self.convert_exp(updated_exps)

            yield x_, a_, y_, r_, reward_sum

            reward_sum = 0
            experiences = [experiences[-1]]

    def run(self):
        # randomly sleep up to 1 second. helps agents boot smoothly.
        time.sleep(np.random.rand())
        np.random.seed(np.int32(time.time() % 1 * 1000 + self.id * 10))

        while self.exit_flag.value == 0:

            total_reward = 0
            total_length = 0
            for x_, a_, y_, r_, reward_sum in self.run_episode():
                total_reward += reward_sum
                total_length += len(r_) + 1  # +1 for last frame that we drop
                self.training_q.put((x_[0], a_[0], y_[0], r_[0]))
            self.episode_log_q.put((datetime.now(), self.role, self.id, total_reward, total_length))
            time.sleep(2)
