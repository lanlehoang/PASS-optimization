"""
Define classes to be used in the env.py file
"""

import heapq  # Used for event priority queue
from typing import List, Optional, TypedDict
import numpy as np
from enum import Enum


class Waveguide:
    def __init__(self, n_pinches, l_wg, h_wg):
        self.n_pinches = n_pinches
        x_pinches = np.linspace(0, l_wg, n_pinches)
        y_pinches = np.zeros(n_pinches)
        z_pinches = np.ones(n_pinches)*h_wg
        self.pinches = np.stack([x_pinches, y_pinches, z_pinches], axis=0)
        self.cur_pinch = 0  # Init at 0 always
        self.antenna = np.array([0, 0, h_wg])

    def configure_pinch(self, target):
        self.cur_pinch = target


class Device:
    def __init__(self, location: np.ndarray):
        self.location = location
        self.has_packet = False # No packet in buffer



class Packet:
    def __init__(self, generation_time):
        self.generation_time = generation_time
        self.end_time = None  # To be set when processing is complete
        self.sent_time = None  # To be set when packet is sent
        self.dropped = False

    def record_end_time(self, end_time):
        """
        Record the end time of the packet processing.
        """
        self.end_time = end_time

    def record_sent_time(self, sent_time):
        """
        Record the sent time of the packet.
        """
        self.sent_time = sent_time

    def drop(self):
        """
        Mark the packet as dropped.
        """
        self.dropped = True


class Experience(TypedDict):
    """
    A single experience tuple (s, a, r, s', done).
    Fields can be None if not yet available.
    """

    state: Optional[np.ndarray]
    action: Optional[int]
    reward: Optional[float]
    next_state: Optional[np.ndarray]
    info: Optional[str]


class ExperienceBuffer:
    """
    Store incomplete experiences e = (s, a, r, s', info).
    Necessary because s' IS NOT IMMEDIATELY AVAILABLE after taking action a.
    The next time the same packet is processed, we can update s' and push it out of the buffer.
    """

    def __init__(self):
        self.buffer = {}
        self.complete_experiences = []

    def add_experience(self, packet_id, experience: Experience):
        """
        Add an experience to the buffer.
        """
        self.buffer[packet_id] = experience

    def update_experience(self, packet_id, new_experience: Experience):
        """
        Update the state of an experience in the buffer based on packet_id.
        """
        if packet_id in self.buffer:
            experience = self.buffer[packet_id]
            for key in new_experience:
                if new_experience[key] is not None:
                    experience[key] = new_experience[key]
        else:
            raise KeyError(f"Packet ID {packet_id} not found in buffer.")

    def complete_experience(self, packet_id):
        """
        Remove a complete experience from the buffer based on packet_id.
        An experience is said to be complete if all of its fields are not None.
        """
        if packet_id in self.buffer:
            self.complete_experiences.append(self.buffer.pop(packet_id))
        else:
            raise KeyError(f"Packet ID {packet_id} not found in buffer.")

    def get_all_complete_experiences(self) -> List[Experience]:
        experiences = self.complete_experiences
        self.complete_experiences = []
        return experiences
