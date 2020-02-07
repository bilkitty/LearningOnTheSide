import os
import unittest
import utils
from collections import defaultdict
from qlearning.qlearning import *
from visualise import *
from environments import EnvTypes, EnvWrapperFactory

# Use this to toggle rendering AND console output
MAX_MEMORY_SIZE = 1
TEST_QTABLE_PKL = os.path.join(utils.GetRootProjectPath(), "tests/data/qtable.pkl")
MOCK_QTABLE = utils.LoadFromPickle(TEST_QTABLE_PKL)
M_ACTIONS = 50
N_SBINS = 500


class TestQlearningSteps(unittest.TestCase):
    def setUp(self):
        self.qla = QLearningAgent(maxEpisodes=1, maxEpochs=1000)
        self.testEnvs = [EnvWrapperFactory(EnvTypes.CartPoleEnv),
                         EnvWrapperFactory(EnvTypes.TaxiGridEnv),
                         EnvWrapperFactory(EnvTypes.WindyGridEnv),
                         EnvWrapperFactory(EnvTypes.AcroBotEnv),
                         EnvWrapperFactory(EnvTypes.MountainCarEnv)]

    def tearDown(self):
        for env in self.testEnvs:
            env.Close()

    def test_Initialization(self):
        for env in self.testEnvs:
            m = env.ActionSpaceN()
            n = env.ObservationSpaceN()
            self.qla.Initialize(env, MAX_MEMORY_SIZE, np.array([M_ACTIONS] * m), np.array([N_SBINS] * n))
            self.assertIsNotNone(self.qla.qTable)
            self.assertIsNotNone(self.qla.experiences)
            self.assertIsNotNone(self.qla.experiences)

        e0 = self.testEnvs[0]
        m = e0.ActionSpaceN()
        n = e0.ObservationSpaceN()
        # Pre-loaded table
        self.qla.Initialize(e0, MAX_MEMORY_SIZE, np.array([M_ACTIONS] * m), np.array([N_SBINS] * n), MOCK_QTABLE)
        # Discretizations of zero
        self.qla.Initialize(e0, MAX_MEMORY_SIZE, np.array([0] * m), np.array([0] * n))

    def test_GetValue(self):
        for env in self.testEnvs:
            m = env.ActionSpaceN()
            n = env.ObservationSpaceN()
            self.qla.Initialize(env, MAX_MEMORY_SIZE, np.array([M_ACTIONS] * m), np.array([N_SBINS] * n))
            s = env.Reset()
            a = self.qla.GetBestAction(s)
            self.assertIsNotNone(self.qla.GetValue(s, a))

    def test_GetBestAction(self):
        for env in self.testEnvs:
            m = env.ActionSpaceN()
            n = env.ObservationSpaceN()
            self.qla.Initialize(env, MAX_MEMORY_SIZE, np.array([M_ACTIONS] * m), np.array([N_SBINS] * n))
            s = env.Reset()
            a = self.qla.GetBestAction(s)
            env.Step(a)
            qa = self.qla.GetValue(s, a)
            qb = self.qla.GetValue(s, self.qla.GetBestAction(s))
            self.assertEqual(qa, qb)

    def test_GetAction(self):
        for env in self.testEnvs:
            m = env.ActionSpaceN()
            n = env.ObservationSpaceN()
            self.qla.Initialize(env, MAX_MEMORY_SIZE, np.array([M_ACTIONS] * m), np.array([N_SBINS] * n))
            a = self.qla.GetAction(env.Reset())
            env.Step(a)
            # We don't do a test for parity of actions or action values b/c
            # this test would likely break for small discrete action spaces


if __name__ == "__main__":
    unittest.main()
