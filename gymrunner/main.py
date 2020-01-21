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
MAX_EPISODES = 100000
MAX_EPOCHS = 100000
LEARNING_RATE = 0.1
DISCOUNT_RATE = 0.6
EPSILON = 0.1
# TODO: understand how this hidden size should be set. Seems env dependent (specifically, action space)
NN_HIDDEN_SIZE = 1
BATCH_SIZE = 128

ENVS = [EnvTypes.WindyGridEnv, EnvTypes.TaxiGridEnv, EnvTypes.CartPoleEnv,
        EnvTypes.AcroBotEnv, EnvTypes.MountainCarEnv, EnvTypes.ContinuousPendulumEnv,
        EnvTypes.ContinuousMountainCarEnv]

ALGOS = ["qlearning", "ddpg"]


def main():
    """
    Currently does not support cartpole environment because the state is used as an index
    into array. Need to use different data structure that'll support different state reps,
    but also be friendly for visualization.
    """
    envIndex = 0
    algoIndex = 0
    maxEpisodes = MAX_EPISODES
    if len(sys.argv) > 1 and sys.argv[1].isdigit():
        envIndex = int(sys.argv[1])
        if envIndex < 0 or len(ENVS) <= envIndex:
            print(f"Invalid env selected '{envIndex}'")
            return 1
    if len(sys.argv) > 2 and sys.argv[2].isdigit():
        algoIndex = int(sys.argv[2])
        if algoIndex < 0 or len(ALGOS) <= algoIndex:
            print(f"Invalid algo selected '{algoIndex}'")
            return 1
    if len(sys.argv) > 3:
        maxEpisodes = int(sys.argv[3])

    algoType = ALGOS[algoIndex]
    env = EnvWrapperFactory(ENVS[envIndex])
    env.Reset()

    qTableFile = f"{ENVS[envIndex]}_qtable.pkl"
    if algoType.lower() == "bruteforce":
        print("Bruteforcing it")
        print("Finished search")
        return 0
    elif algoType.lower() == "qlearning":
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
            SaveAsPickle(resultsTrain, f"{algoType}_{ENVS[envIndex]}_train.pkl")
            print(f"Finished training: {globalRuntime: .4f}s")
            resultsTest, globalRuntime = agent.Evaluate(env, verbose=VERBOSE)

        SaveAsPickle(resultsTest, f"{algoType}_{ENVS[envIndex]}_test.pkl")
        print(f"Finished evaluation: {globalRuntime: .4f}s")
    elif algoType.lower() == "a2c":
        # TODO: pipe in a2c
        print("nothing to see here...")
    elif algoType.lower() == "ddpg":
        discountRate = 0.9
        hiddenSize = 128
        batchSize = 128
        memoryRate = 1e-2
        maxEpochs = 500 if ENVS[envIndex] == EnvTypes.ContinuousPendulumEnv else MAX_EPOCHS
        print(f"DDPG\nParameters:\n  env={ENVS[envIndex]}\n  episodes={maxEpisodes}\n  epochs={maxEpochs}")
        print(f"\n  hiddenLayerSize={hiddenSize}\n  gamma={discountRate}\n  batchSize={batchSize}")
        ddpga = DdpgAgent(maxMemorySize=5000, maxEpisodes=maxEpisodes, maxEpochs=maxEpochs)
        resultsTrain, globalRuntime = ddpga.Train(env,
                                                  gamma=discountRate,
                                                  tau=memoryRate,
                                                  hiddenSize=hiddenSize,
                                                  actorLearningRate=1e-4,
                                                  criticLearningRate=1e-4,
                                                  batchSize=batchSize,
                                                  verbose=VERBOSE)
        print(f"Finished training: {globalRuntime: .4f}s")
        SaveAsPickle(resultsTrain, f"{algoType}_{ENVS[envIndex]}_train.pkl")
        resultsTest, globalRuntime = ddpga.Test(env, verbose=VERBOSE)
        print(f"Finished testing: {globalRuntime: .4f}s")
        SaveAsPickle(resultsTest, f"{algoType}_{ENVS[envIndex]}_test.pkl")
    else:
        print(f"Unsupported algo type '{algoType}'")
        return 1

    # Good practice to close env when finished :)
    env.Close()

    if not SHOULD_PLOT:
        return 0

    figs = []
    figs.append(PlotPerformanceResults(resultsTrain, env.ActionSpaceLabels(shouldUseShorthand=True), f"{algoType}_{ENVS[envIndex]} training results"))
    figs.append(PlotPerformanceResults(resultsTest, env.ActionSpaceLabels(shouldUseShorthand=True), f"{algoType}_{ENVS[envIndex]} test results"))

    for f in figs:
        SaveFigure(f)


if __name__ == "__main__":
    main()
