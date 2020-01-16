import sys, os

from utils import *
from environments import EnvTypes, EnvWrapperFactory
from visualise import PlotPerformanceResults, SaveFigure

# TODO: python args or consider adding parameter file (prefer latter)


def main():
    env = EnvWrapperFactory(EnvTypes.TaxiGridEnv)

    resultsTrain = LoadFromPickle("train.pkl")
    resultsTest = LoadFromPickle("eval.pkl")

    figs = []
    figs.append(PlotPerformanceResults(resultsTrain, env.ActionSpaceLabels(), "training taxi", xMax=1000))
    figs.append(PlotPerformanceResults(resultsTest, env.ActionSpaceLabels(), "test taxi", xMax=1000))

    for f in figs:
        SaveFigure(f)


if __name__ == "__main__":
    main()
