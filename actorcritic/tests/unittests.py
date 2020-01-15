import unittest
import os
from environments import EnvTypes, EnvWrapperFactory
from ddpg import *

# Use this to toggle rendering AND console output
VERBOSE = False
I_TIMEOUT=100
NN_HIDDEN_SIZE = 3


class TestDdpgComponents(unittest.TestCase):
    def test_CreateAgent(self):
        env = EnvWrapperFactory(EnvTypes.CartPoleEnv)
        self.assertNotEqual(DdpgAgent(env, NN_HIDDEN_SIZE), None)


if __name__ == "__main__":
    unittest.main()