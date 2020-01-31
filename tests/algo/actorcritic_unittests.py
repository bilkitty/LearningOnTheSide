import unittest
import numpy as np
from environments import EnvTypes, EnvWrapperFactory
from actorcritic.ddpg import *

# Use this to toggle rendering AND console output
RENDERING_MODE = "ansi"
I_TIMEOUT = 5
MAX_MEMORY_SIZE = 10
NN_HIDDEN_SIZE = np.array([10, 10])
ALR = 1e-4
CLR = 1e-4
NOISE_THETA = 0.15
NOISE_SIGMA = 0.2
BATCH_SIZE = 2

# TODO: how would this work with discrete envs?


class TestDdpgComponents(unittest.TestCase):
    def setUp(self):
        self.testEnvs = [EnvWrapperFactory(EnvTypes.ContinuousPendulumEnv, renderingMode=RENDERING_MODE),
                         EnvWrapperFactory(EnvTypes.ContinuousMountainCarEnv, renderingMode=RENDERING_MODE)]
        self.ddpga = DdpgAgent(maxEpisodes=1, maxEpochs=I_TIMEOUT, gamma=0.9, tau=0.5, batchSize=BATCH_SIZE)

    def tearDown(self):
        for env in self.testEnvs:
            env.Close()

    def test_SetupNetworks(self):
        for env in self.testEnvs:
            self.ddpga.SetupNetworks(env, NN_HIDDEN_SIZE)
            self.assertIsNotNone(self.ddpga.actor)
            self.assertIsNotNone(self.ddpga.actorTarget)
            self.assertIsNotNone(self.ddpga.critic)
            self.assertIsNotNone(self.ddpga.criticTarget)
            for src, tar in zip(self.ddpga.actor.parameters(), self.ddpga.actorTarget.parameters()):
                self.assertEqual(torch.sum(torch.eq(src, tar)), src.nelement())
            for src, tar in zip(self.ddpga.critic.parameters(), self.ddpga.criticTarget.parameters()):
                self.assertEqual(torch.sum(torch.eq(src, tar)), src.nelement())

    def test_SetOptimizers(self):
        for env in self.testEnvs:
            agent = DdpgAgent(maxEpisodes=1, maxEpochs=I_TIMEOUT, gamma=0.9, tau=0.5, batchSize=BATCH_SIZE)
            self.assertFalse(agent.SetupOptimizers(actorLearningRate=1, criticLearningRate=1))
            agent.SetupNetworks(env, NN_HIDDEN_SIZE)
            self.assertTrue(agent.SetupOptimizers(actorLearningRate=1, criticLearningRate=1))

    def test_SetupNoiseProcess(self):
        for env in self.testEnvs:
            self.ddpga.SetupNoiseProcess(env)
            self.assertIsNotNone(self.ddpga.noiseProcess)

    def test_TakeAction(self):
        for env in self.testEnvs:
            self.ddpga.Initialize(env, MAX_MEMORY_SIZE, NN_HIDDEN_SIZE, ALR, CLR, NOISE_SIGMA, NOISE_THETA)
            a = self.ddpga.GetAction(env.Reset())
            env.Step(a)

    def test_TakeActionNoNoise(self):
        for env in self.testEnvs:
            self.ddpga.Initialize(env, MAX_MEMORY_SIZE, NN_HIDDEN_SIZE, ALR, CLR, NOISE_SIGMA, NOISE_THETA)
            # Note that for ddpg, a noiseless action and best action are one in the same
            a = self.ddpga.GetBestAction(env.Reset())
            env.Step(a)

    def test_RunUpdate(self):
        for env in self.testEnvs:
            self.ddpga.Initialize(env, MAX_MEMORY_SIZE, NN_HIDDEN_SIZE, ALR, CLR, NOISE_SIGMA, NOISE_THETA)

            s = env.Reset()
            a = self.ddpga.GetAction(s)
            ns, r, done, _ = env.Step(a)
            self.ddpga.SaveExperience(state=s, action=a, nextState=ns, done=done, reward=r)
            self.ddpga.SaveExperience(state=s, action=a, nextState=ns, done=done, reward=r)
            self.ddpga.SaveExperience(state=s, action=a, nextState=ns, done=done, reward=r)
            self.assertTrue(self.ddpga.Update())

    def test_RunUpdateWithoutExperience(self):
        for env in self.testEnvs:
            self.ddpga.Initialize(env, MAX_MEMORY_SIZE, NN_HIDDEN_SIZE, ALR, CLR, NOISE_SIGMA, NOISE_THETA)
            self.assertFalse(self.ddpga.Update())

    def test_GetValue(self):
        for env in self.testEnvs:
            self.ddpga.Initialize(env, MAX_MEMORY_SIZE, NN_HIDDEN_SIZE, ALR, CLR, NOISE_SIGMA, NOISE_THETA)
            s = env.Reset()
            self.assertIsNotNone(self.ddpga.GetValue(s, self.ddpga.GetAction(s)))


if __name__ == "__main__":
    unittest.main()