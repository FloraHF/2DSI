import tensorflow as tf
from Actor import Actor
from Critic import Critic
from Config import Config
import math

class NetworkAC(object):
    """docstring for NetworkAC."""
    def __init__(self):
        tf.reset_default_graph()
        self.sess = tf.Session()
        self.actor = Actor(self.sess, \
                        n_features=Config.PLAYER_DIMENSION*(Config.DEFENDER_COUNT+Config.INTRUDER_COUNT), \
                        lr=Config.LEARNING_RATE_START, action_bound=[-math.pi, math.pi])
        self.critic = Critic(self.sess, \
                        n_features=Config.PLAYER_DIMENSION*(Config.DEFENDER_COUNT+Config.INTRUDER_COUNT), \
                        lr=Config.LEARNING_RATE_START)
        self.sess.run(tf.global_variables_initializer())

    def train(self, x, a, y, r):
        td_error = self.critic.learn(x, r, y)  # gradient = grad[r + gamma * V(y_) - V(x_)]
        self.actor.learn(x, a, td_error)  # true_gradient = grad[logPi(s,a) * td_error]

    def predict(self, state):
        action = self.actor.choose_action(state)
        value = self.critic.predict(state)
        return action, value
