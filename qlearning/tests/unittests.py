import unittest
import utils
from qlearning import *
from environments import EnvTypes, ENV_DICTIONARY

TEST_QTABLE_PKL = "data/test.pkl"
VERBOSE = False
qtable = utils.LoadFromPickle(TEST_QTABLE_PKL)


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
        qla.SetParameters(maxEpisodes=1, maxEpochs=1)
        results, time = qla.Train(env, lambda x: x + 1, verbose=VERBOSE)
        self.assertEqual(len(results), 1)
        for i, res in enumerate(results):
            self.assertNotEqual(res, None, msg=f"result {i} is 'None'")

    def SingleTestCycle(self, env):
        qla = QLearningAgent()
        qla.SetParameters(maxEpisodes=1, maxEpochs=1)
        results, time = qla.Evaluate(env, qtable)
        self.assertEqual(len(results), 1)
        for i, res in enumerate(results):
            self.assertNotEqual(res, None, msg=f"result {i} is 'None'")

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


if __name__ == "__main__":
    unittest.main()
