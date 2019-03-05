from Config import Config
from WorldMap import WorldMap
from ProcessStats import ProcessStats
from ProcessIntruder import ProcessIntruder
from ProcessDefender import ProcessDefender

from multiprocessing import Process, Array, Value, Queue
from ctypes import Structure, c_double, c_int
import time

class state_structure(Structure):
     _fields_ = [('x', c_double), ('y', c_double), ('done', c_int)]


class EnvironmentMDMI():
    """docstring for GameEnvironment."""
    def __init__(self):

        # environment parameters
        self.world = WorldMap()
        # number of defenders and intruders
        self.dcount = Config.DEFENDER_COUNT
        self.icount = Config.INTRUDER_COUNT

        self.init_dstates = []
        self.init_istates = []
        for d in range(self.dcount):
            self.init_dstates.append((-5+8*d, 7-d, 0))
        for i in range(self.icount):
            self.init_istates.append((5-2*i, 10+i, 0))

        self.dstates = Array(state_structure, self.init_dstates)
        self.istates = Array(state_structure, self.init_istates)

        self.done = Value('i', 0)
        self.stats = ProcessStats(self)

        # defenders and intruders
        self.defenders = [] # all the defender objectives
        self.intruders = []
        for i in range(self.icount):
            self.intruders.append(ProcessIntruder(self, i, self.istates[i], self.stats.episode_log_q))
            self.intruders[i].daemon = True
        for d in range(self.dcount):
            self.defenders.append(ProcessDefender(self, d, self.dstates[d], self.stats.episode_log_q))
            self.defenders[d].daemon = True

    def reset(self):
        # reset states
        for d in range(self.dcount):
            self.defenders[d].state.x = self.init_dstates[d][0]
            self.defenders[d].state.y = self.init_dstates[d][1]
        for i in range(self.icount):
            self.intruders[i].state.x = self.init_istates[i][0]
            self.intruders[i].state.y = self.init_istates[i][1]
        # reset done
        for d in range(self.dcount):
            self.defenders[d].state.done = self.init_dstates[d][2]
            self.defenders[d].capture_buffer.value = 0
            self.defenders[d].enter_buffer.value = 0
        for i in range(self.icount):
            self.intruders[i].state.done = self.init_istates[i][2]
            self.intruders[i].captured.value = 0
            self.intruders[i].entered.value = 0

    def is_game_done(self):
        done = 1
        for d in range(self.dcount):
            done *= self.defenders[d].state.done
        for i in range(self.icount):
            done *= self.intruders[i].state.done
        self.done.value = done
        return self.done.value

    def save_model(self):
        for d in range(self.dcount):
            self.defenders[d].model.save()
        for i in range(self.icount):
            self.intruders[i].model.save()

    def main(self):
        # start the processes
        self.stats.start()
        for i in range(self.icount):
            self.intruders[i].start()
            self.intruders[i].predictor.start()
            self.intruders[i].trainer.start()
        for d in range(self.dcount):
            self.defenders[d].start()
            self.defenders[d].predictor.start()
            self.defenders[d].trainer.start()

        # reset environment when all players are done, untill the max episode is reached
        while self.stats.episode_count.value < Config.EPISODES:
            # print(self.stats.episode_count)
            # print('I am running, intruder.done = ', self.intruders[0].state.done)
            if self.is_game_done():
                self.reset()

        # quit all the processes
        for d in range(self.dcount):
            self.defenders[d].exit_flag.value = 1
            self.defenders[d].join()
        for i in range(self.icount):
            self.intruders[i].exit_flag.value = 1
            self.intruders[i].join()
        self.stats.exit_flag.value = 1
        self.stats.join()
