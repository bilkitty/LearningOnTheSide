import sys, os

from utils import *
from actorcritic.ddpg import DdpgAgent
from qlearning.qlearning import QLearningAgent
from environments import EnvTypes, EnvWrapperFactory
from visualise import PlotPerformanceResults, SaveFigure

SHOULD_REUSE_QTABLE = False
SHOULD_PLOT = True
VERBOSE = False

# TODO: python args or consider adding parameter file (prefer latter)
ALGO_TYPE = "qlearning"
MAX_EPISODES = 100000
MAX_EPOCHS = 100000
LEARNING_RATE = 0.6
DISCOUNT_RATE = 0.1
EPSILON = 0.1
NN_HIDDEN_SIZE = 3

ENVS = [EnvTypes.WindyGridEnv, EnvTypes.TaxiGridEnv, EnvTypes.CartPoleEnv,
        EnvTypes.AcroBotEnv, EnvTypes.MountainCarEnv]


def main():
    """
    Currently does not support cartpole environment because the state is used as an index
    into array. Need to use different data structure that'll support different state reps,
    but also be friendly for visualization.
    """
    envIndex = 0
    maxEpisodes = MAX_EPISODES
    if len(sys.argv) > 1 and sys.argv[1].isdigit():
        envIndex = int(sys.argv[1])
        if envIndex < 0 or len(ENVS) <= envIndex:
            print(f"Invalid env selected '{envIndex}'")
            return 1
    if len(sys.argv) > 2:
        maxEpisodes = int(sys.argv[2])

    env = EnvWrapperFactory(ENVS[envIndex])
    env.Reset()

    qTableFile = f"{ENVS[envIndex]}_qtable.pkl"
    if ALGO_TYPE.lower() == "bruteforce":
        print("Bruteforcing it")
        print("Finished search")
        return 0
    elif ALGO_TYPE.lower() == "qlearning":
        print(f"Q-learning it\nParameters:\n  env={ENVS[envIndex]}\n  episodes={maxEpisodes}\n  epochs={MAX_EPOCHS}")
        print(f"\n  epsilon={EPSILON}\n  gamma={DISCOUNT_RATE}\n  alpha={LEARNING_RATE}\n")
        agent = QLearningAgent()
        agent.SetParameters(EPSILON, DISCOUNT_RATE, LEARNING_RATE, maxEpisodes, MAX_EPOCHS)
        policy = agent.CreatePolicyFunction()
        # load q-table if available
        if SHOULD_REUSE_QTABLE and os.path.exists(qTableFile):
            qTable = LoadFromPickle(qTableFile)
            print("Loaded q-table")
            resultsTrain = None
            resultsTest, globalRuntime = agent.Evaluate(env, qTable, verbose=VERBOSE)
        else:
            resultsTrain, globalRuntime = agent.Train(env, policy, verbose=VERBOSE)
            qTable = agent.QValues()
            SaveAsPickle(qTable, qTableFile)
            SaveAsPickle(resultsTrain, f"{ENVS[envIndex]}_train.pkl")
            print(f"Finished training: {globalRuntime: .4f}s")
            resultsTest, globalRuntime = agent.Evaluate(env, verbose=VERBOSE)

        SaveAsPickle(resultsTest, f"{ENVS[envIndex]}_test.pkl")
        print(f"Finished evaluation: {globalRuntime: .4f}s")
    elif ALGO_TYPE.lower() == "ddpg":
        #agent = DdpgAgent(env, NN_HIDDEN_SIZE)
        print("nothing to see here...")
    else:
        print(f"Unsupported algo type '{ALGO_TYPE}'")
        return 1

    # Good practice to close env when finished :)
    env.Close()

    if not SHOULD_PLOT:
        return 0

    figs = []
    figs.append(PlotPerformanceResults(resultsTrain, env.ActionSpaceLabels(shouldUseShorthand=True), f"{ENVS[envIndex]} training results"))
    figs.append(PlotPerformanceResults(resultsTest, env.ActionSpaceLabels(shouldUseShorthand=True), f"{ENVS[envIndex]} test results"))

    for f in figs:
        SaveFigure(f)


if __name__ == "__main__":
    main()
