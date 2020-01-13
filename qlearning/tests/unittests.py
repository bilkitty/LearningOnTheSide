import unittest
import utils
import os
from qlearning import *
from environments import EnvTypes, ENV_DICTIONARY
from collections import defaultdict

VERBOSE = True
I_TIMEOUT=100
TEST_QTABLE_PKL = os.path.join(utils.GetScriptPath(), "data/test.pkl")
QTABLE = utils.LoadFromPickle(TEST_QTABLE_PKL)


class TestEnvironmentCreation(unittest.TestCase):
    def test_CreateWindyGrid(self):
        self.assertNotEqual(ENV_DICTIONARY[EnvTypes.WindyGridEnv](), None)

    def test_CreateTaxiGrid(self):
        self.assertNotEqual(ENV_DICTIONARY[EnvTypes.TaxiGridEnv](), None)

    def test_CreateCartPole(self):
        self.assertNotEqual(ENV_DICTIONARY[EnvTypes.CartPoleEnv](), None)


class TestQlearningSteps(unittest.TestCase):

    def SingleTrainCycle(self, env):
        qla = QLearningAgent()
        qla.SetParameters(maxEpisodes=1, maxEpochs=10000)
        results, time = qla.Train(env, lambda x: x + 1, verbose=VERBOSE)
        self.assertEqual(len(results), 1)
        for i, res in enumerate(results):
            self.assertNotEqual(res, None, msg=f"result {i} is 'None'")

        env.close()

    def SingleTestCycle(self, env):
        qla = QLearningAgent()
        qla.SetParameters(maxEpisodes=1, maxEpochs=10000)
        results, time = qla.Evaluate(env, QTABLE, verbose=VERBOSE)
        self.assertEqual(len(results), 1)
        for i, res in enumerate(results):
            self.assertNotEqual(res, None, msg=f"result {i} is 'None'")

        env.close()

    def test_SingleTrainCycleOnWindy(self):
        self.SingleTrainCycle(ENV_DICTIONARY[EnvTypes.WindyGridEnv]())

    def test_SingleTrainCycleOnTaxi(self):
        self.SingleTrainCycle(ENV_DICTIONARY[EnvTypes.TaxiGridEnv]())

    def test_SingleTrainCycleOnCartPole(self):
        self.SingleTrainCycle(ENV_DICTIONARY[EnvTypes.CartPoleEnv]())

    def test_SingleEvaluationCycleOnWindy(self):
        self.SingleTestCycle(ENV_DICTIONARY[EnvTypes.WindyGridEnv]())

    def test_SingleEvaluationCycleOnTaxi(self):
        self.SingleTestCycle(ENV_DICTIONARY[EnvTypes.TaxiGridEnv]())

    def test_SingleEvaluationCycleOnCartPole(self):
        self.SingleTestCycle(ENV_DICTIONARY[EnvTypes.CartPoleEnv]())

    def test_CreatePolicyFunction(self):
        # Setup mock q-table
        abest = 0
        state = "s0"
        mockqtable = defaultdict(lambda: np.zeros(3))
        mockqtable[state][abest] = 1

        # Setup agent that never explores
        qla = QLearningAgent()
        qla.SetParameters(epsilon=0, maxEpisodes=1, maxEpochs=1)
        purelyExploitPolicy = qla.CreatePolicyFunction(mockqtable)
        a = purelyExploitPolicy(state)
        self.assertEqual(a, abest)

        # Setup agent that never exploits
        qla.SetParameters(epsilon=1, maxEpisodes=1, maxEpochs=1)
        purelyExplorePolicy = qla.CreatePolicyFunction(mockqtable)
        a = purelyExplorePolicy(state)
        # betting that best action isn't chosen multiple in a row?
        for i in np.arange(I_TIMEOUT):
            a = purelyExplorePolicy(state)
            if a != abest or I_TIMEOUT < i:
                break

        self.assertNotEqual(a, abest)


if __name__ == "__main__":
    unittest.main()
