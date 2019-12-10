#!/usr/bin/python3

import gym

env = gym.make("Taxi-v3").env

env.render()