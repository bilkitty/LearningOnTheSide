import torch
import unittest
from environments import EnvTypes, EnvWrapperFactory
from actorcritic.ddpg import *

# Use this to toggle rendering AND console output
VERBOSE = False
I_TIMEOUT=100
NN_HIDDEN_SIZE = 3

# TODO: how would this work with discrete envs?

class TestDdpgComponents(unittest.TestCase):

    def test_SetupNetworks(self):
        env = EnvWrapperFactory(EnvTypes.ContinuousPendulumEnv)
        ddpga = DdpgAgent(maxMemorySize=1)
        ddpga.SetupNetworks(env, NN_HIDDEN_SIZE)
        self.assertIsNotNone(ddpga.actor)
        self.assertIsNotNone(ddpga.actorTarget)
        self.assertIsNotNone(ddpga.critic)
        self.assertIsNotNone(ddpga.criticTarget)
        for src, tar in zip(ddpga.actor.parameters(), ddpga.actorTarget.parameters()):
            self.assertEqual(torch.sum(torch.eq(src, tar)), src.nelement())
        for src, tar in zip(ddpga.critic.parameters(), ddpga.criticTarget.parameters()):
            self.assertEqual(torch.sum(torch.eq(src, tar)), src.nelement())

    def test_SetOptimizers(self):
        ddpga = DdpgAgent(maxMemorySize=1)
        self.assertFalse(ddpga.SetupOptimizers(actorLearningRate=1, criticLearningRate=1))
        ddpga.SetupNetworks(EnvWrapperFactory(EnvTypes.CartPoleEnv), NN_HIDDEN_SIZE)
        self.assertTrue(ddpga.SetupOptimizers(actorLearningRate=1, criticLearningRate=1))

    def test_Experiences(self):
        ddpga = DdpgAgent(maxMemorySize=1)


    def test_Train(self):
        env = EnvWrapperFactory(EnvTypes.CartPoleEnv)
        ddpga = DdpgAgent(maxMemorySize=1, maxEpisodes=1, maxEpochs=1)
        ddpga.Train(env, 0.6, NN_HIDDEN_SIZE, 1e-4, 1e-4, 1)
        env.Close()


if __name__ == "__main__":
    unittest.main()