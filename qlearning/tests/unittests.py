import unittest
from  utils import *
from qlearning import *
from environments import EnvTypes, ENV_DICTIONARY

TEST_QTABLE_PKL = "test.pkl"
# TODO: no idea why these are not running... followed doc almost exactly...

class TestEnvironmentCreation(unittest.TestCase):
    def CreateWindyGrid(self):
        self.assertNotEqual(ENV_DICTIONARY[EnvTypes.WindyGridEnv](), None)

    def CreateTaxiGrid(self):
        self.assertNotEqual(ENV_DICTIONARY[EnvTypes.TaxiGridEnv](), None)

    def CreateCartPole(self):
        self.assertNotEqual(ENV_DICTIONARY[EnvTypes.CartPoleEnv](), None)


class TestQlearningSteps(unittest.TestCase):
    def SingleTrainCycle(self):
        # check results are available
        # check success
        self.assertTrue(False)

    def SingleEvaluationCycle(self):
        # load qtable
        # check results are available
        # check success
        self.assertTrue(False)


if __name__ == "__main__":
    unittest.main()
