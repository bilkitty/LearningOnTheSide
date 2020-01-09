import sys, os

from utils import *
from qlearning import QLearningAgent
from environments import EnvTypes, ENV_DICTIONARY

QTABLE_FILE = "qtable.pkl"

# TODO: python args or consider adding parameter file (prefer latter)
ALGO_TYPE = "qlearning"
MAX_EPISODES = 1
MAX_EPOCHS = 1
LEARNING_RATE = 0.1
DISCOUNT_RATE = 0.6
EPSILON = 0.1

def main():
    env = ENV_DICTIONARY[EnvTypes.TaxiGridEnv]()
    agent = QLearningAgent()
    policy = agent.CreatePolicy()
    agent.SetParameters(EPSILON, DISCOUNT_RATE, LEARNING_RATE, MAX_EPISODES, MAX_EPOCHS)
    env.reset()

    infoText = f"state space: {env.action_space}\n"
    infoText += f"obs space: {env.observation_space}\n"

    resultsTrain = None
    succeeded = False
    totalTrainingTime = totalEvaluationTime = 0
    if (ALGO_TYPE.lower() == "bruteforce"):
        print("Bruteforcing it")
        print("Finished search")
        return
    elif (ALGO_TYPE.lower() == "qlearning"):
        print("Q-learning it")
        # load qtable if available
        if (os.path.exists(QTABLE_FILE)):
            qtable = LoadFromPickle(QTABLE_FILE)
            print("Loaded policy")
        else:
            globalRuntime, resultsTrain = agent.Train(env, policy)
            qtable = agent.QValues()
            SaveAsPickle(qtable, QTABLE_FILE)
            SaveAsPickle(resultsTrain, "train.pkl")
            print(f"Finished training: {globalRuntime: .4f}s")

        assert qtable is not None, "No qtable available"
        globalRuntime, resultsTest = agent.Evaluate(env, policy)
        SaveAsPickle(resultsTest, "eval.pkl")
        print(f"Finished evaluation: {globalRuntime: .4f}s")
    else:
        print(f"Unsupported algo type {ALGO_TYPE}")
        return


if __name__ == "__main__":
    main()
