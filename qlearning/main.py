#!/usr/bin/python3

"""
This example was taken from https://www.geeksforgeeks.org/q-learning-in-python/.
"""

import gym
import itertools 
import matplotlib 
import matplotlib.style 
import numpy as np 
import pandas as pd 
import os
import sys 
  
  
from collections import defaultdict 
from windy_gridworld import WindyGridWorldEnv 
import plotting 

matplotlib.style.use("ggplot")

windyGridEnv = WindyGridWorldEnv()
nEpisodes = 10

EPSILON = 0.1       # prob of exploit/explore
ALPHA = 0.1	        # learning rate
GAMMA = 0.6	        # reward discount rate

""""
Summary of required packages:
python3 -m pip install gym, pandas, numpy, matplotlib

"""


# Epsilon greedy algo: with prob (1-eps), choose action 
# which maximizes q-val, otherwise choose random action. 
def createEpsilonGreedyPolicy(Q, epsilon, num_actions): 
    """
    Creates an epsilon-greedy policy based
    on a given Q-function and epsilon.

    Returns a function that takes the state
    as an input and returns the probabilities
    for each action in the form of a numpy array
    of length of the action space(set of possible actions).
    """
    def policyFunction(state):

        # This is later used by numpy random choice which will select from
        # this array with a non-uniform distribution defined by the array
        # elements itself. In other words, the best action will be favored
        # or discouraged depending on epsilon.
        Action_probabilities = np.ones(num_actions, dtype = float) * epsilon / num_actions

        best_action = np.argmax(Q[state])
        Action_probabilities[best_action] += (1.0 - epsilon)
        return Action_probabilities

    return policyFunction

# Q learning model
def qLearning(env, num_episodes, verbose_mode, discount_factor = 1.0, 
                            alpha = 0.6, epsilon = 0.1):
    """
    Q-Learning algorithm: Off-policy TD control.
    Finds the optimal greedy policy while improving
    following an epsilon-greedy policy"""

    # Action value function
    # A nested dictionary that maps
    # state -> (action -> action-value).
    Q = defaultdict(lambda: np.zeros(env.action_space.n))

    # Keeps track of useful statistics
    stats = plotting.EpisodeStats(
        episode_lengths = np.zeros(num_episodes),
        episode_rewards = np.zeros(num_episodes))

    # Create an epsilon greedy policy function
    # appropriately for environment action space
    policy = createEpsilonGreedyPolicy(Q, epsilon, env.action_space.n)

    # For every episode
    for ith_episode in range(num_episodes):

        # Reset the environment and pick the first action
        state = env.reset()

        for t in itertools.count():

            # get probabilities of all actions from current state
            action_probabilities = policy(state)

            # choose action according to
            # the probability distribution
            action = np.random.choice(np.arange(
                    len(action_probabilities)),
                    p = action_probabilities)

            # take action and get reward, transit to next state
            next_state, reward, done, _ = env.step(action)

            # Update statistics
            stats.episode_rewards[ith_episode] += reward
            stats.episode_lengths[ith_episode] = t

            # TD Update
            # TODO: is this correct? seems like old qval should be scaled by 1-alpha, but
            #       maybe it's all the same.
            best_next_action = np.argmax(Q[next_state])
            td_target = reward + discount_factor * Q[next_state][best_next_action]
            td_delta = td_target - Q[state][action]
            Q[state][action] += alpha * td_delta

            if verbose_mode:
                env.render("episode: ({},{})  rewards:{}".format(
                    ith_episode, t, stats.episode_rewards[ith_episode]))

            # done is True if episode terminated
            if done:
                break

            state = next_state

    return Q, stats

#
# Experiments
#

# basic training
outputPath = ""
actionVal, stats = qLearning(windyGridEnv, nEpisodes, True, discount_factor=GAMMA, alpha=ALPHA, epsilon=EPSILON)
fig = plotting.plot_episode_stats(stats, smoothing_window = nEpisodes // 10)
fig.savefig(os.path.join(outputPath, "basic_training_results.png"))

print("la fin")