import os
import unittest
import utils
from visualise import *
from metrics import Metrics
from environments import EnvTypes, EnvWrapperFactory
from noiseprocess import OUStrategy
from baseargs import BaseArgsParser

TEST_METRICS_PKL = os.path.join(utils.GetRootProjectPath(), "tests/data/mountaincar_metrics.pkl")
TEST_ARGS_JSON = os.path.join(utils.GetRootProjectPath(), "tests/data/testargs.json")
MOCK_RESULTS = Metrics.LoadMetricsFromPickle(TEST_METRICS_PKL)
MOCK_ARGS_CMDLN = ["--envIndex", "0", "--verbose"]


class TestVisualisation(unittest.TestCase):

    def test_PerformancePlot(self):
        env = EnvWrapperFactory(EnvTypes.MountainCarEnv, renderingMode="ansi")
        fig = PlotPerformanceResults(MOCK_RESULTS, env.ActionSpaceLabels(shouldUseShorthand=True), "test mtn car")
        self.assertTrue(fig.get_axes())
        fig.savefig("test_PerformancePlot.png")


# NOTE: use continuous action space environments for these tests
class TestNoiseProcess(unittest.TestCase):
    def setUp(self):
        self.envWrapper = EnvWrapperFactory(EnvTypes.ContinuousMountainCarEnv, renderingMode="ansi")

    def tearDown(self):
        self.envWrapper.Close()

    def test_OrnsteinUhlenbeckCreation(self):
        self.assertIsNotNone(OUStrategy(self.envWrapper.env.action_space))

    def test_OrnsteinUhlenbeckGetAction(self):
        noiseModel = OUStrategy(self.envWrapper.env.action_space)
        randomAction = self.envWrapper.env.action_space.sample()
        self.assertIsNotNone(noiseModel.get_action(randomAction))
        self.assertNotEqual(noiseModel.get_action(randomAction), randomAction)


class TestArgsParser(unittest.TestCase):
    def setUp(self):
        with open(TEST_ARGS_JSON, "wb") as f:
            f.write("{\"arg0\": 1, \"arg1\": 42.4242, \"arg3\": \"type0\"}".encode())
            f.close()

        self.parser = BaseArgsParser("test args")

    def test_ParseArgs(self):
        args = self.parser.ParseArgs(TEST_ARGS_JSON, MOCK_ARGS_CMDLN)
        self.assertIsNotNone(args.arg0)
        self.assertIsNotNone(args.arg1)
        self.assertIsNotNone(args.envIndex)

    def test_GetJsonArgs(self):
        argDict = self.parser.GetJsonArgs(TEST_ARGS_JSON)
        self.assertEqual(argDict["arg0"], 1)
        self.assertEqual(argDict["arg1"], 42.4242)
        self.assertEqual(argDict["arg3"], "type0")


if __name__ == "__main__":
    unittest.main()
