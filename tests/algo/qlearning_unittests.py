import os
import unittest
import utils
from collections import defaultdict
from qlearning.qlearning import *
from visualise import *
from environments import EnvTypes, EnvWrapperFactory

# Use this to toggle rendering AND console output
VERBOSE = False
I_TIMEOUT=100
# TODO: set env variables for proj dir, test dir, etc.
TEST_QTABLE_PKL = os.path.join(utils.GetScriptPath(), "../../data/test.pkl")
QTABLE = utils.LoadFromPickle(TEST_QTABLE_PKL)


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
        self.SingleTrainCycle(EnvWrapperFactory(EnvTypes.WindyGridEnv))

    def test_SingleTrainCycleOnTaxi(self):
        self.SingleTrainCycle(EnvWrapperFactory(EnvTypes.TaxiGridEnv))

    def test_SingleTrainCycleOnCartPole(self):
        self.SingleTrainCycle(EnvWrapperFactory(EnvTypes.CartPoleEnv))

    def test_SingleTrainCycleOnAcroBot(self):
        self.SingleTrainCycle(EnvWrapperFactory(EnvTypes.AcroBotEnv))

    def test_SingleTrainCycleOnMountainCar(self):
        self.SingleTrainCycle(EnvWrapperFactory(EnvTypes.MountainCarEnv))

    def test_SingleEvaluationCycleOnWindy(self):
        self.SingleTestCycle(EnvWrapperFactory(EnvTypes.WindyGridEnv))

    def test_SingleEvaluationCycleOnTaxi(self):
        self.SingleTestCycle(EnvWrapperFactory(EnvTypes.TaxiGridEnv))

    def test_SingleEvaluationCycleOnCartPole(self):
        self.SingleTestCycle(EnvWrapperFactory(EnvTypes.CartPoleEnv))

    def test_SingleEvaluationCycleOnAcroBot(self):
        self.SingleTestCycle(EnvWrapperFactory(EnvTypes.AcroBotEnv))

    def test_SingleEvaluationCycleOnMountainCar(self):
        self.SingleTestCycle(EnvWrapperFactory(EnvTypes.MountainCarEnv))

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
