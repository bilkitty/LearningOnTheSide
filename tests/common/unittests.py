import os
import unittest
import utils
from visualise import *
from metrics import Metrics
from environments import EnvTypes, EnvWrapperFactory
from noiseprocess import OUStrategy
from baseargs import BaseArgsParser
from memory import Memory

TEST_METRICS_PKL = os.path.join(utils.GetRootProjectPath(), "tests/data/mountaincar_metrics.pkl")
TEST_ARGS_JSON = os.path.join(utils.GetRootProjectPath(), "tests/data/testargs.json")
MOCK_RESULTS = Metrics.LoadMetricsFromPickle(TEST_METRICS_PKL)
MOCK_ARGS_CMDLN = ["--envIndex", "0", "--verbose"]
MOCK_ITEM = 42
MAX_MEMORY_SIZE = 10


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

class TestMemoryBuffer(unittest.TestCase):
    def setUp(self):
        self.memory = Memory(MAX_MEMORY_SIZE)

    def test_Push(self):
        self.memory.reset()
        self.memory.push(state=MOCK_ITEM, action=MOCK_ITEM, next_state=MOCK_ITEM, reward=MOCK_ITEM, done=MOCK_ITEM)
        self.assertEqual(self.memory.__len__(), 1)

        self.memory.push(state=MOCK_ITEM, action=MOCK_ITEM, next_state=MOCK_ITEM, reward=MOCK_ITEM, done=MOCK_ITEM)
        self.memory.push(state=MOCK_ITEM, action=MOCK_ITEM, next_state=MOCK_ITEM, reward=MOCK_ITEM, done=MOCK_ITEM)
        self.assertEqual(self.memory.__len__(), 3)

        for i in range(0, MAX_MEMORY_SIZE - 1):
            self.memory.push(state=MOCK_ITEM, action=MOCK_ITEM, next_state=MOCK_ITEM, reward=MOCK_ITEM, done=MOCK_ITEM)
        self.assertEqual(self.memory.__len__(), MAX_MEMORY_SIZE)

    def test_Sample(self):
        self.memory.reset()
        i0, i1, i2, i3, i4 = self.memory.sample(1)
        self.assertEqual(len(i0), 0)
        self.assertEqual(len(i1), 0)
        self.assertEqual(len(i2), 0)
        self.assertEqual(len(i3), 0)
        self.assertEqual(len(i4), 0)

        self.memory.push(state=MOCK_ITEM, action=MOCK_ITEM, next_state=MOCK_ITEM, reward=MOCK_ITEM, done=MOCK_ITEM)
        self.memory.push(state=MOCK_ITEM, action=MOCK_ITEM, next_state=MOCK_ITEM, reward=MOCK_ITEM, done=MOCK_ITEM)
        self.memory.push(state=MOCK_ITEM, action=MOCK_ITEM, next_state=MOCK_ITEM, reward=MOCK_ITEM, done=MOCK_ITEM)
        i0, i1, i2, i3, i4 = self.memory.sample(2)
        self.assertEqual(len(i0), 2)
        self.assertEqual(len(i1), 2)
        self.assertEqual(len(i2), 2)
        self.assertEqual(len(i3), 2)
        self.assertEqual(len(i4), 2)

    def test_GetLastest(self):
        self.memory.reset()
        self.memory.push(state=MOCK_ITEM, action=MOCK_ITEM, next_state=MOCK_ITEM, reward=MOCK_ITEM, done=MOCK_ITEM)
        self.memory.push(state=0, action=0, next_state=0, reward=0, done=0)
        self.assertEqual(self.memory.getLatest()[0], 0)
        self.assertEqual(self.memory.getLatest()[0], MOCK_ITEM)


class TestArgsParser(unittest.TestCase):
    def setUp(self):
        with open(TEST_ARGS_JSON, "wb") as f:
            f.write("{\"arg0\": 1, \"arg1\": 42.4242, \"arg3\": \"type0\"}".encode())
            f.close()

        self.parser = BaseArgsParser()

    def test_ParseArgs(self):
        args = self.parser.ParseArgs(TEST_ARGS_JSON, MOCK_ARGS_CMDLN)
        self.assertIsNotNone(args.arg0)
        self.assertIsNotNone(args.arg1)
        self.assertIsNotNone(args.envIndex)
        self.assertEqual(args.arg0, 1)
        self.assertEqual(args.arg1, 42.4242)
        self.assertEqual(args.arg3, "type0")


if __name__ == "__main__":
    unittest.main()
