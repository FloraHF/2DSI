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

import numpy as np
import math

class Config:

    #########################################################################
    # Game configuration

    # Time step size
    TIME_STEP = 0.2

    # The world boundary, as a square
    WORLD_X_BOUND = 10
    WORLD_Y_BOUND = 10

    # Target area, a circle for now (TODO: define another class specifying different
    # types of target area, and specify the type name here)

    # Target type
    TARGET_TYPE = 'circle'

    # Radius of the target area
    TARGET_RADIUS = 5
    # Center of the target area
    TARGET_CENTER = [0, 0]

    # Number of defenders
    DEFENDER_COUNT = 2
    # Number of intruders
    INTRUDER_COUNT = 1

    # Dynamic types
    DEFENDER_DYNAMIC = 'simple_motion'
    INTRUDER_DYNAMIC = 'simple_motion'
    # Action spaces
    DEFENDER_ACTION_SPACE = [-math.pi, math.pi]
    INTRUDER_ACTION_SPACE = [-math.pi, math.pi]
    # Maximum velocities
    DEFENDER_MAX_VELOCITY = 1
    INTRUDER_MAX_VELOCITY = 1.5
    # Maximum accelerations
    DEFENDER_MAX_ACCELERATION = 1
    INTRUDER_MAX_ACCELERATION = 1
    # Defender's capture range
    CAPTURE_RANGE = 2

    # Reward definitions
    PENALTY_TIME_PASS = - 0.05 * TIME_STEP
    REWARD_CAPTURE = 0.9
    REWARD_ENTER = 0.9

    #########################################################################
    # Game configuration

    # Enable to see the trained agent in action
    PLAY_MODE = False
    # Enable to train
    TRAIN_MODELS = True
    # Load old models. Throws if the model doesn't exist
    LOAD_CHECKPOINT = False
    # If 0, the latest checkpoint is loaded
    LOAD_EPISODE = 0

    #########################################################################
    # Device
    DEVICE = '/cpu:0'

    #########################################################################
    # Algorithm parameters

    # Discount factor
    DISCOUNT = 0.99

    # Tmax
    TIME_MAX = 5

    # Reward Clipping
    REWARD_MIN = -1
    REWARD_MAX = 1

    # Max size of the queue
    MAX_QUEUE_SIZE = 100
    PREDICTION_BATCH_SIZE = 128

    # Input of the DNN
    STACKED_FRAMES = 1
    PLAYER_COUNT = DEFENDER_COUNT + INTRUDER_COUNT
    PLAYER_DIMENSION = 3

    # Total number of episodes and annealing frequency
    EPISODES = 62
    DEFENDER_ANNEALING_EPISODE_COUNT = 400000
    INTRUDER_ANNEALING_EPISODE_COUNT = 400000

    # Entropy regualrization hyper-parameter
    BETA_START = 0.01
    BETA_END = 0.01

    # Learning rate
    LEARNING_RATE_START = 0.0003
    LEARNING_RATE_END = 0.0003

    # RMSProp parameters
    RMSPROP_DECAY = 0.99
    RMSPROP_MOMENTUM = 0.0
    RMSPROP_EPSILON = 0.1

    # Dual RMSProp - we found that using a single RMSProp for the two cost function works better and faster
    DUAL_RMSPROP = False

    # Gradient clipping
    USE_GRAD_CLIP = False
    GRAD_CLIP_NORM = 40.0
    # Epsilon (regularize policy lag in GA3C)
    LOG_EPSILON = 1e-6
    # Training min batch size - increasing the batch size increases the stability of the algorithm, but make learning slower
    TRAINING_MIN_BATCH_SIZE = 0

    #########################################################################
    # Log and save

    # Enable TensorBoard
    TENSORBOARD = False
    # Update TensorBoard every X training steps
    TENSORBOARD_UPDATE_FREQUENCY = 1000

    # Enable to save models every SAVE_FREQUENCY episodes
    SAVE_MODELS = True
    # Save every SAVE_FREQUENCY episodes
    SAVE_FREQUENCY = 1

    # Print stats every PRINT_STATS_FREQUENCY episodes
    PRINT_STATS_FREQUENCY = 1
    # The window to average stats
    STAT_ROLLING_MEAN_WINDOW = 1000

    # Results filename
    RESULTS_FILENAME = 'results.txt'
    TRAJECTORY_FILENAME = 'trj_'
    # Network checkpoint name
    NETWORK_NAME = ''

    #########################################################################
    # More experimental parameters here

    # Minimum policy
    MIN_POLICY = 0.0
    # Use log_softmax() instead of log(softmax())
    USE_LOG_SOFTMAX = False
