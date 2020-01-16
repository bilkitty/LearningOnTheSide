import os
import unittest
import utils
from visualise import *
from metrics import Metrics
from environments import EnvTypes, EnvWrapperFactory

# TODO: set env variables for proj dir, test dir, etc.
TEST_METRICS_PKL = os.path.join(utils.GetScriptPath(), "../data/mountaincar_metrics.pkl")
MOCK_RESULTS = Metrics.LoadMetricsFromPickle(TEST_METRICS_PKL)


class TestVisualisation(unittest.TestCase):
    def test_PerformancePlot(self):
        env = EnvWrapperFactory(EnvTypes.MountainCarEnv, renderingMode="ansi")
        fig = PlotPerformanceResults(MOCK_RESULTS, env.ActionSpaceLabels(shouldUseShorthand=True), "test mtn car")
        self.assertTrue(fig.get_axes())
        fig.savefig("test_PerformancePlot.png")


# TODO: add test for memory
#   - what happens if buffersize > items in queue?

if __name__ == "__main__":
    unittest.main()
