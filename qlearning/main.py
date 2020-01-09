import sys, os

from utils import *
from qlearning import QLearningAgent
from environments import EnvTypes, ENV_DICTIONARY

QTABLE_FILE = "qtable.pkl"
ALGO_TYPE = "qlearning"

# TODO: python args

def main():
    env = ENV_DICTIONARY[EnvTypes.CartPoleEnv]()
    agent = QLearningAgent()
    policy = agent.CreatePolicy()
    agent.SetParameters()
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
            succeeded, resultsTrain = agent.Train(env, policy)
            qtable = agent.QValues()
            SaveAsPickle(qtable, QTABLE_FILE)
            SaveAsPickle(resultsTrain, "train.pkl")
            print(f"Finished training: {resultsTrain.globalRuntime: .4f}s")

        assert qtable is not None, "No qtable available"
        succeeded, resultsTest = agent.Evaluate(env, policy)
        SaveAsPickle(resultsTest, "eval.pkl")
        print(f"Finished evaluation: {resultsTest.globalRuntime: .4f}s")
    else:
        print(f"Unsupported algo type {ALGO_TYPE}")
        return


if __name__ == "__main__":
    main()
