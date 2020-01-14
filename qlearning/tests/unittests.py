import unittest
import utils
import os
from collections import defaultdict
from qlearning import *
from environments import EnvTypes, ENV_DICTIONARY
from visualise import *

# Use this to toggle rendering AND console output
VERBOSE = False
I_TIMEOUT=100
#TEST_QTABLE_PKL = os.path.join(utils.GetScriptPath(), "data/test.pkl")
TEST_QTABLE_PKL = "data/test.pkl"
TEST_METRICS_PKL = "data/test_metrics.pkl"
QTABLE = utils.LoadFromPickle(TEST_QTABLE_PKL)
MOCK_RESULTS = utils.LoadFromPickle(TEST_METRICS_PKL)


class TestEnvironmentCreation(unittest.TestCase):
    def test_CreateWindyGrid(self):
        self.assertNotEqual(ENV_DICTIONARY[EnvTypes.WindyGridEnv](), None)

    def test_CreateTaxiGrid(self):
        self.assertNotEqual(ENV_DICTIONARY[EnvTypes.TaxiGridEnv](), None)

    def test_CreateCartPole(self):
        self.assertNotEqual(ENV_DICTIONARY[EnvTypes.CartPoleEnv](), None)

    def test_CreateAcroBot(self):
        self.assertNotEqual(ENV_DICTIONARY[EnvTypes.AcroBotEnv](), None)

    def test_CreateMountainCar(self):
        self.assertNotEqual(ENV_DICTIONARY[EnvTypes.MountainCarEnv](), None)


class TestQlearningSteps(unittest.TestCase):

    def SingleTrainCycle(self, env):
        qla = QLearningAgent()
        qla.SetParameters(maxEpisodes=1, maxEpochs=10000)
        policy = qla.CreatePolicyFunction()
        results, time = qla.Train(env, policy, verbose=VERBOSE)
        self.assertEqual(len(results), 1)
        for i, res in enumerate(results):
            self.assertNotEqual(res, None, msg=f"result {i} is 'None'")

        env.Close()

    def SingleTestCycle(self, env):
        qla = QLearningAgent()
        qla.SetParameters(maxEpisodes=1, maxEpochs=10000)
        results, time = qla.Evaluate(env, QTABLE, verbose=VERBOSE)
        self.assertEqual(len(results), 1)
        for i, res in enumerate(results):
            self.assertNotEqual(res, None, msg=f"result {i} is 'None'")

        env.Close()

    def test_SingleTrainCycleOnWindy(self):
        self.SingleTrainCycle(ENV_DICTIONARY[EnvTypes.WindyGridEnv]())

    def test_SingleTrainCycleOnTaxi(self):
        self.SingleTrainCycle(ENV_DICTIONARY[EnvTypes.TaxiGridEnv]())

    def test_SingleTrainCycleOnCartPole(self):
        self.SingleTrainCycle(ENV_DICTIONARY[EnvTypes.CartPoleEnv]())

    def test_SingleTrainCycleOnAcroBot(self):
        self.SingleTrainCycle(ENV_DICTIONARY[EnvTypes.AcroBotEnv]())

    def test_SingleTrainCycleOnMountainCar(self):
        self.SingleTrainCycle(ENV_DICTIONARY[EnvTypes.MountainCarEnv]())

    def test_SingleEvaluationCycleOnWindy(self):
        self.SingleTestCycle(ENV_DICTIONARY[EnvTypes.WindyGridEnv]())

    def test_SingleEvaluationCycleOnTaxi(self):
        self.SingleTestCycle(ENV_DICTIONARY[EnvTypes.TaxiGridEnv]())

    def test_SingleEvaluationCycleOnCartPole(self):
        self.SingleTestCycle(ENV_DICTIONARY[EnvTypes.CartPoleEnv]())

    def test_SingleEvaluationCycleOnAcroBot(self):
        self.SingleTestCycle(ENV_DICTIONARY[EnvTypes.AcroBotEnv]())

    def test_SingleEvaluationCycleOnMountainCar(self):
        self.SingleTestCycle(ENV_DICTIONARY[EnvTypes.MountainCarEnv]())

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

class TestVisualisation(unittest.TestCase):
    def test_PerformancePlot(self):
            

if __name__ == "__main__":
    unittest.main()
