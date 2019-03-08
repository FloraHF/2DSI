# Copyright (c) 2016, NVIDIA CORPORATION. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#  * Neither the name of NVIDIA CORPORATION nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
# OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import os
import re
import numpy as np
import tensorflow as tf

from Config import Config

class NetworkHH(object):
    def __init__(self, device, model_name, abound):
        self.device = device
        self.model_name = model_name
        self.abound = abound

        self.pcount = Config.DEFENDER_COUNT + Config.INTRUDER_COUNT
        self.p_dim = Config.PLAYER_DIMENSION

        self.learning_rate = Config.LEARNING_RATE_START
        self.beta = Config.BETA_START
        self.log_epsilon = Config.LOG_EPSILON

        self.graph = tf.Graph()
        with self.graph.as_default() as g:
            # with tf.device(self.device):
            self._create_graph()
            self.sess = tf.Session()

            # self.sess = tf.Session(
            #     graph=self.graph,
            #     config=tf.ConfigProto(
            #         allow_soft_placement=True,
            #         log_device_placement=False,
            #         gpu_options=tf.GPUOptions(allow_growth=True)))
            self.sess.run(tf.global_variables_initializer())

            if Config.LOAD_CHECKPOINT or Config.SAVE_MODELS:
                vars = tf.global_variables()
                self.saver = tf.train.Saver({var.name: var for var in vars}, max_to_keep=0)

    def _create_graph(self):
        # placeholders for samples
        self.x = tf.placeholder(
            tf.float32, [None, self.p_dim*self.pcount], name='X')
        self.a = tf.placeholder(tf.float32, [None, 1], name="Act")
        self.y_r = tf.placeholder(tf.float32, [None], name='Yr')

        self.var_beta = tf.placeholder(tf.float32, name='beta', shape=[])
        self.var_learning_rate = tf.placeholder(tf.float32, name='lr', shape=[])

        self.global_step = tf.Variable(0, trainable=False, name='step')

        # actor-critic network, actor
        l_a = self.dense_layer(self.x, 200, 'la', tf.nn.relu6)
        mu  = self.dense_layer(l_a, 1, 'mu', tf.nn.tanh)
        sigma = self.dense_layer(l_a, 1, 'sigma', tf.nn.softplus)
        mu, sigma = mu * self.abound[1], sigma + 1e-4
        a_dist = tf.contrib.distributions.Normal(mu, sigma)
        self.A = tf.clip_by_value(tf.squeeze(a_dist.sample(1), axis=0), self.abound[0], self.abound[1])[0, :]
        # select action
        # actor-critic network, critic
        l_c = self.dense_layer(self.x, 100, 'lc', tf.nn.relu6)
        self.v = tf.squeeze(self.dense_layer(l_c, 1, 'v', func=None), axis=[1])

        # loss of critic
        td = self.y_r - self.v
        self.c_loss = 0.5 * tf.reduce_sum(tf.square(td), axis=0)
        # loss of actor
        a_loss_1 = a_dist.log_prob(self.a) * td
        a_loss_2 = self.var_beta * a_dist.entropy()
        self.a_loss = tf.reduce_mean(-(a_loss_1 + a_loss_2))

        # set optimizer
        if Config.DUAL_RMSPROP:
            self.opt_a = tf.train.RMSPropOptimizer(
                learning_rate=self.var_learning_rate,
                decay=Config.RMSPROP_DECAY,
                momentum=Config.RMSPROP_MOMENTUM,
                epsilon=Config.RMSPROP_EPSILON)
            self.opt_c = tf.train.RMSPropOptimizer(
                learning_rate=self.var_learning_rate,
                decay=Config.RMSPROP_DECAY,
                momentum=Config.RMSPROP_MOMENTUM,
                epsilon=Config.RMSPROP_EPSILON)
        else:
            self.loss_all = self.a_loss + self.c_loss
            self.opt = tf.train.RMSPropOptimizer(
                learning_rate=self.var_learning_rate,
                decay=Config.RMSPROP_DECAY,
                momentum=Config.RMSPROP_MOMENTUM,
                epsilon=Config.RMSPROP_EPSILON)

        if Config.USE_GRAD_CLIP:
            if Config.DUAL_RMSPROP:
                self.opt_grad_c = self.opt_c.compute_gradients(self.c_loss)
                self.opt_grad_c_clipped = [(tf.clip_by_norm(g, Config.GRAD_CLIP_NORM),v)
                                            for g,v in self.opt_grad_c if not g is None]
                self.train_op_c = self.opt_c.apply_gradients(self.opt_grad_c_clipped)

                self.opt_grad_a = self.opt_a.compute_gradients(self.a_loss)
                self.opt_grad_a_clipped = [(tf.clip_by_norm(g, Config.GRAD_CLIP_NORM),v)
                                            for g,v in self.opt_grad_a if not g is None]
                self.train_op_a = self.opt_a.apply_gradients(self.opt_grad_a_clipped)
                self.train_op = [self.train_op_a, self.train_op_c]
            else:
                self.opt_grad = self.opt.compute_gradients(self.loss_all)
                self.opt_grad_clipped = [(tf.clip_by_average_norm(g, Config.GRAD_CLIP_NORM),v) for g,v in self.opt_grad]
                self.train_op = self.opt.apply_gradients(self.opt_grad_clipped)
        else:
            if Config.DUAL_RMSPROP:
                self.train_op_c = self.opt_c.minimize(self.c_loss, global_step=self.global_step)
                self.train_op_a = self.opt_a.minimize(self.a_loss, global_step=self.global_step)
                self.train_op = [self.train_op_a, self.train_op_c]
            else:
                self.train_op = self.opt.minimize(self.loss_all, global_step=self.global_step)

    def dense_layer(self, input, out_dim, name, func=tf.nn.relu):
        in_dim = input.get_shape().as_list()[-1]
        d = 1.0 / np.sqrt(in_dim)
        with tf.variable_scope(name):
            w_init = tf.random_uniform_initializer(-d, d)
            b_init = tf.random_uniform_initializer(-d, d)
            w = tf.get_variable('w', dtype=tf.float32, shape=[in_dim, out_dim], initializer=w_init)
            b = tf.get_variable('b', shape=[out_dim], initializer=b_init)
            output = tf.matmul(input, w) + b
            if func is not None:
                output = func(output)
        return output

    def __get_base_feed_dict(self):
        return {self.var_beta: self.beta, self.var_learning_rate: self.learning_rate}

    def get_global_step(self):
        step = self.sess.run(self.global_step)
        return step

    def predict_v(self, x):
        value = self.sess.run(self.v, feed_dict={self.x: x})
        return value[0]

    def predict_a(self, x):
        action = self.sess.run(self.A, feed_dict={self.x: x})
        return action

    def train(self, x, a, y_r):
        feed_dict = self.__get_base_feed_dict()
        feed_dict.update({self.x: x, self.a: a, self.y_r: y_r})
        self.sess.run(self.train_op, feed_dict=feed_dict)

    def _checkpoint_filename(self, episode):
        return 'checkpoints/%s_%08d' % (self.model_name, episode)

    def _get_episode_from_filename(self, filename):
        # TODO: hacky way of getting the episode. ideally episode should be stored as a TF variable
        return int(re.split('/|_|\.', filename)[2])

    def save(self, episode):
        name = self._checkpoint_filename(episode)
        print(name)
        self.saver.save(self.sess, name)

    def load(self):
        filename = tf.train.latest_checkpoint(os.path.dirname(self._checkpoint_filename(episode=0)))
        if Config.LOAD_EPISODE > 0:
            filename = self._checkpoint_filename(Config.LOAD_EPISODE)
        self.saver.restore(self.sess, filename)
        return self._get_episode_from_filename(filename)

    def get_variables_names(self):
        return [var.name for var in self.graph.get_collection('trainable_variables')]

    def get_variable_value(self, name):
        return self.sess.run(self.graph.get_tensor_by_name(name))
