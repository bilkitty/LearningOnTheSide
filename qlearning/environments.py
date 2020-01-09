import gym
import sys
import os
import random
import numpy as np
from time import sleep
from gym.envs.toy_text import discrete

"""
 Environment generator
"""
class EnvTypes:
    WindyGridEnv = "WindyGridWorld"
    TaxiGridEnv = "TaxiGridWorld"
    CartPoleEnv = "CartPole"

ENV_DICTIONARY = {
    "WindyGridWorld": lambda : WindyGridWorldEnv(),
    "TaxiGridWorld": lambda : gym.make("Taxi-v3").env,
    "CartPole": lambda : gym.make('CartPole-v0')
}

"""
 Cart pole
 
 An inverted pendulum balancing task for a pole attached to a cart that can move
 horizontally along a track. 
"""

"""
 Taxi grid world
 
 A taxi must pick up a passenger from one of 4 fixed
pick/drop locations, traverse a 5x5 grid, and drop them off at one of the
other pick/drop locations. The taxi performs one of the following actions at
a time: pickup, dropoff, move left, right, up, or down. The target pickup and
dropoff locations are marked as blue and purple, respectively. Once a
passenger is picked up from the correct location (requiring two time steps)
the taxi color will change to green. An attempt to pickup or drop off the
passenger at the wrong location will incur -10 reward. Also, each move incurs
-1 reward thus encouraging the taxi to minimize time. When the passenger is in
the taxi and a pickup (incorrectly) occurs, then the env only gives a time penalty.
The agent receives reward of 20 for successfully dropping off the passenger.

Action space
    There are six total possible actions. Obviously, only one action is taken at a time: 
    4 directions of movement + pickup + dropoff.
    

State space
    There are 500 total possible states, including when the passenger is in the car: 
    5x5 grid positions x (4 + 1) passenger positions x 4 destinations 
    Really, each row is a summarization of the car's and passenger's possible position in the grid. It 
    doesn't clearly map to a particular visual representation of the grid. Or maybe we just need to find the
    appropriate mapping.

Notes:  This reward structure seems to allow repeated pickup of the passenger
after they are in the taxi. This situation be an example of how an over
simplified reward signal might allow the system to waste resources, in this case time. 
"""

"""
 Windy grid world
 
 A custom env...
"""

class WindyGridWorldEnv(discrete.DiscreteEnv):

    UP = 0
    RIGHT = 1
    DOWN = 2
    LEFT = 3
    metadata = {'render.modes': ['human', 'ansi']}
    startPos = (3, 0)
    targetPos = (3, 7)

    def __init__(self):
        self.shape = (7, 10)

        self.startPos = (random.randint(0, self.shape[0]-1), random.randint(0, self.shape[1]-1))

        nS = np.prod(self.shape)
        nA = 4

        # Wind strength
        winds = np.zeros(self.shape)
        winds[:,[3,4,5,8]] = 1
        winds[:,[6,7]] = 2

        # Calculate transition probabilities
        P = {}
        for s in range(nS):
            position = np.unravel_index(s, self.shape)
            P[s] = { a : [] for a in range(nA) }
            P[s][WindyGridWorldEnv.UP] = self._calculate_transition_prob(position, [-1, 0], winds)
            P[s][WindyGridWorldEnv.RIGHT] = self._calculate_transition_prob(position, [0, 1], winds)
            P[s][WindyGridWorldEnv.DOWN] = self._calculate_transition_prob(position, [1, 0], winds)
            P[s][WindyGridWorldEnv.LEFT] = self._calculate_transition_prob(position, [0, -1], winds)

        # We always start in state (3, 0)
        isd = np.zeros(nS)
        isd[np.ravel_multi_index(self.startPos, self.shape)] = 1.0

        discrete.DiscreteEnv.__init__(self, nS, nA, P, isd)
        #super(WindyGridWorldEnv, self).__init__(nS, nA, P, isd)

    def render(self, heading, mode='human', close=False):
        self._render(heading, mode, close)

    def _limit_coordinates(self, coord):
        coord[0] = min(coord[0], self.shape[0] - 1)
        coord[0] = max(coord[0], 0)
        coord[1] = min(coord[1], self.shape[1] - 1)
        coord[1] = max(coord[1], 0)
        return coord

    def _calculate_transition_prob(self, current, delta, winds):
        new_position = np.array(current) + np.array(delta) + np.array([-1, 0]) * winds[tuple(current)]
        new_position = self._limit_coordinates(new_position).astype(int)
        new_state = np.ravel_multi_index(tuple(new_position), self.shape)
        is_done = tuple(new_position) == self.targetPos
        return [(1.0, new_state, -1.0, is_done)]

    def _render(self, heading, mode='human', close=False):
        if close:
            return

        if mode != 'ansi': # system calls should generally be avoided...
            outfile = sys.stdout
            os.system('cls' if os.name == 'nt' else 'clear')
        else:
            outfile = StringIO()

        outfile.write(heading + '\n')
        for s in range(self.nS):
            position = np.unravel_index(s, self.shape)
            # print(self.s)
            if self.s == s:
                output = " x "
            elif position == (3,7):
                output = " T "
            else:
                output = " o "

            if position[1] == 0:
                output = output.lstrip()
            if position[1] == self.shape[1] - 1:
                output = output.rstrip()
                output += "\n"

            outfile.write(output)
        outfile.write("\n")

        sleep(0.1)