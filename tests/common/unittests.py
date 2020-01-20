import os
import unittest
import utils
from visualise import *
from metrics import Metrics
from environments import EnvTypes, EnvWrapperFactory
from noiseprocess import OUStrategy

# TODO: set env variables for proj dir, test dir, etc.
TEST_METRICS_PKL = os.path.join(utils.GetRootProjectPath(), "tests/data/mountaincar_metrics.pkl")
MOCK_RESULTS = Metrics.LoadMetricsFromPickle(TEST_METRICS_PKL)


class TestVisualisation(unittest.TestCase):
    def test_PerformancePlot(self):
        env = EnvWrapperFactory(EnvTypes.MountainCarEnv, renderingMode="ansi")
        fig = PlotPerformanceResults(MOCK_RESULTS, env.ActionSpaceLabels(shouldUseShorthand=True), "test mtn car")
        self.assertTrue(fig.get_axes())
        fig.savefig("test_PerformancePlot.png")


# TODO: add test for memory
#   - what happens if buffersize > items in queue?

# NOTE: use continuous action space environments for these tests
class TestNoiseProcess(unittest.TestCase):
    def test_ornsteinUhlenbeckCreation(self):
        env = EnvWrapperFactory(EnvTypes.ContinuousPendulumEnv, renderingMode="ansi")
        self.assertIsNotNone(OUStrategy(env.env.action_space))

    def test_ornsteinUhlenbeckGetAction(self):
        env = EnvWrapperFactory(EnvTypes.ContinuousPendulumEnv, renderingMode="ansi")
        noiseModel = OUStrategy(env.env.action_space)
        randomAction = env.env.action_space.sample() # probs doesn't work
        self.assertIsNotNone(noiseModel.get_action(randomAction))
        self.assertNotEqual(noiseModel.get_action(randomAction), randomAction)


if __name__ == "__main__":
    unittest.main()
