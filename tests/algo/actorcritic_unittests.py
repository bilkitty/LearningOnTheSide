import torch
import unittest
from environments import EnvTypes, EnvWrapperFactory
from actorcritic.ddpg import *

# Use this to toggle rendering AND console output
VERBOSE = False
RENDERING_MODE = "ansi"
I_TIMEOUT = 5
NN_HIDDEN_SIZE = 1
BATCH_SIZE = 127

#TODO: keep track of instantiated environments and ensure they are closed before exiting in tearDown()

# TODO: how would this work with discrete envs?


class TestDdpgComponents(unittest.TestCase):
    def setUp(self):
        self.env = EnvWrapperFactory(EnvTypes.ContinuousPendulumEnv, renderingMode=RENDERING_MODE)

    def tearDown(self):
        self.env.Close()

    def test_SetupNetworks(self):
        ddpga = DdpgAgent(maxMemorySize=1)
        ddpga.SetupNetworks(self.env, NN_HIDDEN_SIZE)
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
        ddpga.SetupNetworks(self.env, NN_HIDDEN_SIZE)
        self.assertTrue(ddpga.SetupOptimizers(actorLearningRate=1, criticLearningRate=1))

    def test_TrainOnPendulum(self):
        hiddenLayers = 10 # TODO: Why use single layer for compatibility with this env?
        ddpga = DdpgAgent(maxMemorySize=1, maxEpisodes=I_TIMEOUT)
        ddpga.Train(self.env, 0.6, 1, hiddenLayers, 1e-4, 1e-4, BATCH_SIZE)

    def test_TrainOnMountainCar(self):
        hiddenLayers = 10 # TODO: Why use single layer for compatibility with this env?
        env = EnvWrapperFactory(EnvTypes.ContinuousMountainCarEnv, renderingMode=RENDERING_MODE)
        ddpga = DdpgAgent(maxMemorySize=1, maxEpisodes=I_TIMEOUT)
        exetime = ddpga.Train(env, 0.6, 1, hiddenLayers, 1e-4, 1e-4, BATCH_SIZE)
        print(f"Training time: {exetime}")
        env.Close()

    def test_takeActionWithNoise(self):
        ddpga = DdpgAgent(maxMemorySize=1, maxEpisodes=I_TIMEOUT)
        ddpga.SetupNetworks(self.env, 10)
        ddpga.SetupOptimizers(1e-4, 1e-4)
        ddpga.SetupNoiseProcess(self.env.env)
        action = ddpga.GetAction(self.env.Reset(), 0)
        self.env.Step(action)

    def test_takeActionNoNoise(self):
        ddpga = DdpgAgent(maxMemorySize=1, maxEpisodes=I_TIMEOUT)
        ddpga.SetupNetworks(self.env, 10)
        ddpga.SetupOptimizers(1e-4, 1e-4)
        ddpga.SetupNoiseProcess(self.env.env)
        action = ddpga.GetAction(self.env.Reset(), 0, shouldAddNoise=False)
        self.env.Step(action)


if __name__ == "__main__":
    unittest.main()