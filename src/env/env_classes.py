"""
Define classes to be used in the env.py file
"""

import heapq  # Used for event priority queue
from typing import List, Optional, TypedDict
import numpy as np
from enum import Enum


class Waveguide:
    def __init__(self, n_pinches, l_wg, h_wg, switch_cost=0):
        self.n_pinches = n_pinches
        x_pinches = np.linspace(0, l_wg, n_pinches)
        y_pinches = np.zeros(n_pinches)
        z_pinches = np.ones(n_pinches)*h_wg
        self.pinches = np.stack([x_pinches, y_pinches, z_pinches], axis=0)
        self.cur_pinch = 0  # Init at 0 always
        self.antenna = np.array([0, 0, h_wg])
        self.switch_cost = switch_cost

    def configure_pinch(self, target):
        self.cur_pinch = target

    def switch_pinch(self, target):
        t_switch = self.switch_cost if self.cur_pinch != target else 0
        self.cur_pinch = target
        return t_switch
