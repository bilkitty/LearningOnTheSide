import os
import unittest
import utils
from visualise import *
from metrics import Metrics
from environments import EnvTypes, EnvWrapperFactory
from noiseprocess import OUStrategy
from baseargs import BaseArgsParser

# TODO: set env variables for proj dir, test dir, etc.
TEST_METRICS_PKL = os.path.join(utils.GetRootProjectPath(), "tests/data/mountaincar_metrics.pkl")
TEST_ARGS_JSON = os.path.join(utils.GetRootProjectPath(), "tests/data/tes.parser.json")
MOCK_ARGS_JSON = "{\"arg0\": 1, \"arg1\": 42.4242, \"arg3\": \"type0\"}"
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
    def setUp(self):
        self.env = EnvWrapperFactory(EnvTypes.MountainCarEnv, renderingMode="ansi").env

    def test_ornsteinUhlenbeckCreation(self):
        self.assertIsNotNone(OUStrategy(self.env.action_space))

    def test_ornsteinUhlenbeckGetAction(self):
        noiseModel = OUStrategy(self.env.action_space)
        randomAction = self.env.action_space.sample() # probs doesn't work
        self.assertIsNotNone(noiseModel.get_action(randomAction))
        self.assertNotEqual(noiseModel.get_action(randomAction), randomAction)


class TestArgsParser(unittest.TestCase):
    def setUp(self):
        self.parser = BaseArgsParser("test args")

    def test_cmdlnParser(self):
        argList = self.parser.GetCmdlnArgs()

    def test_jsonParser(self):
        argDict = self.parser.GetJsonArgs(TEST_ARGS_JSON)
        self.assertEqual(argDict["arg0"], 1)
        self.assertEqual(argDict["arg1"], 42.4242)
        self.assertEqual(argDict["arg3"], "type0")


if __name__ == "__main__":
    unittest.main()
