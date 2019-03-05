import sys
if sys.version_info >= (3,0):
    from queue import Queue as queueQueue
else:
    from Queue import Queue as queueQueue
from datetime import datetime
from multiprocessing import Process, Value, Queue


import numpy as np
import time

from ProcessIntruder import ProcessIntruder
from ProcessDefender import ProcessDefender
from Config import Config


class ProcessStats(Process):
    def __init__(self, env):
        super(ProcessStats, self).__init__()
        self.env = env
        self.episode_log_q = Queue(maxsize=100)
        self.episode_count = Value('i', 0)
        self.training_count = Value('i', 0)
        self.should_save_model = Value('i', 0)
        self.exit_flag = Value('i', 0)

    def run(self):
        with open(Config.RESULTS_FILENAME, 'a') as results_logger:

            rolling_reward = 0
            results_q = queueQueue(maxsize=Config.STAT_ROLLING_MEAN_WINDOW)

            self.start_time = time.time()
            first_time = datetime.now()

            while not (self.exit_flag.value and self.episode_log_q.empty()):
    
                episode_time, player, pid, reward, length = self.episode_log_q.get()
                results_logger.write('%s, %s, %s, %d, %d\n' % (episode_time.strftime("%Y-%m-%d %H:%M:%S"), player, pid, reward, length))
                results_logger.flush()

                self.episode_count.value += 1
                rolling_reward += reward

                if results_q.full():
                    old_episode_time, old_player, old_pid, old_reward, old_length = results_q.get()
                    rolling_reward -= old_reward
                    first_time = old_episode_time

                results_q.put((episode_time, player, pid, reward, length))

                if self.episode_count.value % Config.SAVE_FREQUENCY == 0:
                    self.should_save_model.value = 1

                if self.episode_count.value % Config.PRINT_STATS_FREQUENCY == 0:
                    print(
                        '[Time: %8d Episode: %8d] '
                        '[%s %s\'s Reward: %10.4f RRward: %10.4f] '
                        % (int(time.time()-self.start_time), self.episode_count.value,
                           player, pid, reward, rolling_reward/results_q.qsize()))
                    sys.stdout.flush()
