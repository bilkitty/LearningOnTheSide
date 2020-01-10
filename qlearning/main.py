import sys, os

from utils import *
from qlearning import QLearningAgent
from environments import EnvTypes, ENV_DICTIONARY

QTABLE_FILE = "qtable.pkl"
SHOULD_REUSE_QTABLE = False

# TODO: python args or consider adding parameter file (prefer latter)
ALGO_TYPE = "qlearning"
MAX_EPISODES = 1
MAX_EPOCHS = 1
LEARNING_RATE = 0.1
DISCOUNT_RATE = 0.6
EPSILON = 0.1

def main():
    """
    Currently does not support cartpole environment because the state is used as an index
    into array. Need to use different data structure that'll support different state reps,
    but also be friendly for visualization.
    """
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
        if (SHOULD_REUSE_QTABLE and os.path.exists(QTABLE_FILE)):
            qtable = LoadFromPickle(QTABLE_FILE)
            print("Loaded policy")
        else:
            resultsTrain, globalRuntime = agent.Train(env, policy)
            qTable = agent.QValues()
            SaveAsPickle(qTable, QTABLE_FILE)
            SaveAsPickle(resultsTrain, "train.pkl")
            print(f"Finished training: {globalRuntime: .4f}s")

        resultsTest, globalRuntime = agent.Evaluate(env)
        SaveAsPickle(resultsTest, "eval.pkl")
        print(f"Finished evaluation: {globalRuntime: .4f}s")
    else:
        print(f"Unsupported algo type {ALGO_TYPE}")
        return

    # Good practice to close env when finished :)
    env.close()


if __name__ == "__main__":
    main()
