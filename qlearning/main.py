import sys, os

from utils import *
from qlearning import QLearningAgent
from environments import EnvTypes, ENV_DICTIONARY
from visualise import PlotPerformanceResults, SaveFigure

QTABLE_FILE = "qtable.pkl"
SHOULD_REUSE_QTABLE = False
SHOULD_PLOT = True
VERBOSE = False

# TODO: python args or consider adding parameter file (prefer latter)
ALGO_TYPE = "qlearning"
MAX_EPISODES = 10000
MAX_EPOCHS = 100000
LEARNING_RATE = 0.1
DISCOUNT_RATE = 0.6
EPSILON = 0.1


def main():
    """
    Currently does not support cartpole environment because the state is used as an index
    into array. Need to use different data structure that'll support different state reps,
    but also be friendly for visualization.
    """
    env = ENV_DICTIONARY[EnvTypes.MountainCarEnv]()
    agent = QLearningAgent()
    policy = agent.CreatePolicyFunction()
    agent.SetParameters(EPSILON, DISCOUNT_RATE, LEARNING_RATE, MAX_EPISODES, MAX_EPOCHS)
    env.Reset()

    resultsTrain = None
    succeeded = False
    totalTrainingTime = totalEvaluationTime = 0
    if ALGO_TYPE.lower() == "bruteforce":
        print("Bruteforcing it")
        print("Finished search")
        return
    elif ALGO_TYPE.lower() == "qlearning":
        print("Q-learning it")
        # load q-table if available
        if SHOULD_REUSE_QTABLE and os.path.exists(QTABLE_FILE):
            qTable = LoadFromPickle(QTABLE_FILE)
            print("Loaded q-table")
            resultsTest, globalRuntime = agent.Evaluate(env, qTable, verbose=VERBOSE)
        else:
            resultsTrain, globalRuntime = agent.Train(env, policy, verbose=VERBOSE)
            qTable = agent.QValues()
            SaveAsPickle(qTable, QTABLE_FILE)
            SaveAsPickle(resultsTrain, "train.pkl")
            print(f"Finished training: {globalRuntime: .4f}s")
            resultsTest, globalRuntime = agent.Evaluate(env, verbose=VERBOSE)

        SaveAsPickle(resultsTest, "eval.pkl")
        print(f"Finished evaluation: {globalRuntime: .4f}s")
    else:
        print(f"Unsupported algo type {ALGO_TYPE}")
        return

    # Good practice to close env when finished :)
    env.Close()

    if not SHOULD_PLOT:
        return

    figs = []
    figs.append(PlotPerformanceResults(resultsTrain, env.ActionSpaceLabels(shouldUseShorthand=True), "training results"))
    figs.append(PlotPerformanceResults(resultsTest, env.ActionSpaceLabels(shouldUseShorthand=True), "test results"))

    for f in figs:
        SaveFigure(f)


if __name__ == "__main__":
    main()
