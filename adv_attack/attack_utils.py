import torch
from typing import Optional
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

# This class is used to schedule the attack process, it determines when to conduct the attack
class Attack_Scheduler(object):
    def __init__(self, start, step):
        self.start = start  # conduct attack every [start] frames
        self.step = step
        self.counter = 0
        self.need_attack = 1

    def next(self):

        #print("Attack_Scheduler: counter = {}, start = {}, step = {}".format(self.counter, self.start, self.step))

        if self.counter < (self.start + self.step) - 1:
            self.counter += 1
        else:
            self.counter = 0
            self.need_attack = 0  # each sequence is only attack once (but may not one frame)

    def is_attack(self):  # determine whether current frame needs to be attacked

        if self.start == 0:
            return False
        if self.need_attack == 0:
            return False
        if self.counter >= self.start: 
            return True
        else:
            return False

    def attack_counter(self):
        return self.counter

    def reset(self):
        self.counter = 0
        self.need_attack = 1


