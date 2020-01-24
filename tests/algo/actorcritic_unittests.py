import torch
import unittest
from environments import EnvTypes, EnvWrapperFactory
from actorcritic.ddpg import *

# Use this to toggle rendering AND console output
VERBOSE = False
RENDERING_MODE = "ansi"
I_TIMEOUT = 5
MAX_MEMORY_SIZE = 10
NN_HIDDEN_SIZE = 10
ALR = 1e-4
CLR = 1e-4
BATCH_SIZE = 1

#TODO: keep track of instantiated environments and ensure they are closed before exiting in tearDown()

# TODO: how would this work with discrete envs?


class TestDdpgComponents(unittest.TestCase):
    def setUp(self):
        self.env = EnvWrapperFactory(EnvTypes.ContinuousPendulumEnv, renderingMode=RENDERING_MODE)
        self.ddpga = DdpgAgent(maxEpisodes=1, maxEpochs=I_TIMEOUT, gamma=0.9, tau=0.5, batchSize=BATCH_SIZE)

    def tearDown(self):
        self.env.Close()

    def test_SetupNetworks(self):
        self.ddpga.SetupNetworks(self.env, NN_HIDDEN_SIZE)
        self.assertIsNotNone(self.ddpga.actor)
        self.assertIsNotNone(self.ddpga.actorTarget)
        self.assertIsNotNone(self.ddpga.critic)
        self.assertIsNotNone(self.ddpga.criticTarget)
        for src, tar in zip(self.ddpga.actor.parameters(), self.ddpga.actorTarget.parameters()):
            self.assertEqual(torch.sum(torch.eq(src, tar)), src.nelement())
        for src, tar in zip(self.ddpga.critic.parameters(), self.ddpga.criticTarget.parameters()):
            self.assertEqual(torch.sum(torch.eq(src, tar)), src.nelement())

    def test_SetOptimizers(self):
        self.assertFalse(self.ddpga.SetupOptimizers(actorLearningRate=1, criticLearningRate=1))
        self.ddpga.SetupNetworks(self.env, NN_HIDDEN_SIZE)
        self.assertTrue(self.ddpga.SetupOptimizers(actorLearningRate=1, criticLearningRate=1))

    def test_SetupNoiseProcess(self):
        self.ddpga.SetupNoiseProcess(self.env)
        self.assertIsNotNone(self.ddpga.noiseProcess)

    def test_TakeAction(self):
        self.ddpga.Initialize(self.env, MAX_MEMORY_SIZE, NN_HIDDEN_SIZE, ALR, CLR)
        a = self.ddpga.GetAction(self.env.Reset(), shouldAddNoise=True)
        self.env.Step(a)

    def test_TakeActionNoNoise(self):
        self.ddpga.Initialize(self.env, MAX_MEMORY_SIZE, NN_HIDDEN_SIZE, ALR, CLR)
        a = self.ddpga.GetAction(self.env.Reset(), shouldAddNoise=False)
        self.env.Step(a)

    def test_RunUpdate(self):
        self.ddpga.Initialize(self.env, MAX_MEMORY_SIZE, NN_HIDDEN_SIZE, ALR, CLR)

        s = self.env.Reset()
        a = self.ddpga.GetAction(s, shouldAddNoise=False)
        ns, r, done, _ = self.env.Step(a)
        self.ddpga.SaveExperience(state=s, action=a, nextState=ns, done=done, reward=r)
        self.assertTrue(self.ddpga.Update())

    def test_RunUpdateWithoutExperience(self):
        self.ddpga.Initialize(self.env, MAX_MEMORY_SIZE, NN_HIDDEN_SIZE, ALR, CLR)
        self.assertFalse(self.ddpga.Update())

    def test_GetValue(self):
        self.ddpga.Initialize(self.env, MAX_MEMORY_SIZE, NN_HIDDEN_SIZE, ALR, CLR)
        s = self.env.Reset()
        self.assertIsNotNone(self.ddpga.GetValue(s, self.ddpga.GetAction(s, shouldAddNoise=False)))


if __name__ == "__main__":
    unittest.main()