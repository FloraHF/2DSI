import tensorflow as tf
import numpy as np

np.random.seed(2)
tf.set_random_seed(2)  # reproducible


class Actor(object):
    def __init__(self, sess, n_features, action_bound, lr=0.0001):
        self.sess = sess

        self.s = tf.placeholder(tf.float32, [1, n_features], "state")
        self.a = tf.placeholder(tf.float32, None, name="act")
        self.td_error = tf.placeholder(tf.float32, None, name="td_error")  # TD_error

        l1 = tf.layers.dense(
            inputs=self.s,
            units=60,  # number of hidden units
            activation=tf.nn.relu,
            kernel_initializer=tf.random_normal_initializer(0., .1),  # weights
            bias_initializer=tf.constant_initializer(0.1),  # biases
            name='l1'
        )

        l2 = tf.layers.dense(
            inputs=l1,
            units=10,  # number of hidden units
            activation=tf.nn.relu,
            kernel_initializer=tf.random_normal_initializer(0., .1),  # weights
            bias_initializer=tf.constant_initializer(0.1),  # biases
            name='l2'
        )

        l3 = tf.layers.dense(
            inputs=l2,
            units=30,  # number of hidden units
            activation=tf.nn.relu,
            kernel_initializer=tf.random_normal_initializer(0., .1),  # weights
            bias_initializer=tf.constant_initializer(0.1),  # biases
            name='l3'
        )

        mu = tf.layers.dense(
            inputs=l3,
            units=1,  # number of hidden units
            activation=tf.nn.tanh,
            kernel_initializer=tf.random_normal_initializer(0., .1),  # weights
            bias_initializer=tf.constant_initializer(0.1),  # biases
            name='mu'
        )

        sigma = tf.layers.dense(
            inputs=l1,
            units=1,  # output units
            activation=tf.nn.softplus,  # get action probabilities
            kernel_initializer=tf.random_normal_initializer(0., .1),  # weights
            bias_initializer=tf.constant_initializer(1.),  # biases
            name='sigma'
        )
        global_step = tf.Variable(0, trainable=False)
        # self.e = epsilon = tf.train.exponential_decay(2., global_step, 1000, 0.9)
        self.mu, self.sigma = tf.squeeze(mu*2), tf.squeeze(sigma+0.1)
        self.normal_dist = tf.distributions.Normal(self.mu, self.sigma)

        self.action = tf.clip_by_value(self.normal_dist.sample(1), action_bound[0], action_bound[1])

        with tf.name_scope('exp_v'):
            log_prob = self.normal_dist.log_prob(self.a)  # loss without advantage
            self.exp_v = log_prob * self.td_error  # advantage (TD_error) guided loss
            # Add cross entropy cost to encourage exploration
            self.exp_v += 0.01*self.normal_dist.entropy()

        with tf.name_scope('train'):
            self.train_op = tf.train.AdamOptimizer(lr).minimize(-self.exp_v, global_step)    # min(v) = max(-v)

    def learn(self, s, a, td):
        s = s[np.newaxis, :]
        feed_dict = {self.s: s, self.a: a, self.td_error: td}
        _, exp_v = self.sess.run([self.train_op, self.exp_v], feed_dict)
        return exp_v

    def choose_action(self, s):
        s = s[np.newaxis, :]
        return self.sess.run(self.action, {self.s: s})  # get probabilities for all actions
