import sys, os

from utils import *
from environments import EnvTypes, EnvWrapperFactory
from visualise import PlotPerformanceResults, SaveFigure

# TODO: python args or consider adding parameter file (prefer latter)


def main():
    xmax = 500
    algo = "ddpg"
    env = EnvWrapperFactory(EnvTypes.ContinuousPendulumEnv)

    resultsTrain = LoadFromPickle(f"{algo}_{EnvTypes.ContinuousPendulumEnv}_train.pkl")
    resultsTest = LoadFromPickle(f"{algo}_{EnvTypes.ContinuousPendulumEnv}_test.pkl")

    figs = []
    figs.append(PlotPerformanceResults(resultsTrain,
                                       env.ActionSpaceLabels(),
                                       f"training {EnvTypes.ContinuousPendulumEnv}",
                                       xMax=xmax))
    figs.append(PlotPerformanceResults(resultsTest,
                                       env.ActionSpaceLabels(),
                                       f"test {EnvTypes.ContinuousPendulumEnv}",
                                       xMax=xmax))

    for f in figs:
        SaveFigure(f)


if __name__ == "__main__":
    main()
