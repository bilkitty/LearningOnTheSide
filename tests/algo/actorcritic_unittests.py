import torch
import unittest
from environments import EnvTypes, EnvWrapperFactory
from actorcritic.ddpg import *

# Use this to toggle rendering AND console output
VERBOSE = False
I_TIMEOUT = 100
NN_HIDDEN_SIZE = 1
BATCH_SIZE = 127

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
        ddpga.SetupNetworks(EnvWrapperFactory(EnvTypes.ContinuousPendulumEnv), NN_HIDDEN_SIZE)
        self.assertTrue(ddpga.SetupOptimizers(actorLearningRate=1, criticLearningRate=1))

    def test_TrainOnPendulum(self):
        hiddenLayers = 1 # TODO: Why use single layer for compatibility with this env?
        env = EnvWrapperFactory(EnvTypes.ContinuousPendulumEnv)
        ddpga = DdpgAgent(maxMemorySize=1, maxEpisodes=I_TIMEOUT)
        ddpga.Train(env, 0.6, 1, hiddenLayers, 1e-4, 1e-4, BATCH_SIZE)
        env.Close()

    def test_TrainOnMountainCar(self):
        hiddenLayers = 1 # TODO: Why use single layer for compatibility with this env?
        env = EnvWrapperFactory(EnvTypes.ContinuousMountainCarEnv)
        ddpga = DdpgAgent(maxMemorySize=1, maxEpisodes=I_TIMEOUT)
        ddpga.Train(env, 0.6, 1, hiddenLayers, 1e-4, 1e-4, BATCH_SIZE)
        env.Close()


if __name__ == "__main__":
    unittest.main()