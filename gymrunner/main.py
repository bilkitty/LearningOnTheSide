import sys, os
import numpy as np

from utils import *
from baseargs import BaseArgsParser
from actorcritic.ddpg import DdpgAgent
from qlearning.qlearning import QLearningAgent
from environments import EnvTypes, EnvWrapperFactory
from visualise import PlotPerformanceResults, SaveFigure

SHOULD_REUSE_QTABLE = False
SHOULD_PLOT = True

# TODO: python args or consider adding parameter file (prefer latter)
LEARNING_RATE = 0.1
DISCOUNT_RATE = 0.6
EPSILON = 0.1
# TODO: understand how this hidden size should be set. Seems env dependent (specifically, action space)
NN_HIDDEN_SIZE = 1
BATCH_SIZE = 128
PARAM_FILE = os.path.join(GetRootProjectPath(), "gymrunner/params.json")

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
    parser = BaseArgsParser("General algorithm arguments")
    args = parser.ParseArgs(PARAM_FILE)
    envIndex = args.envIndex
    algoIndex = args.algoIndex
    maxEpisodes = args.maxEpisodes
    maxEpochs = args.maxEpochs
    verbose = args.verbose
    postfix = args.desc

    if envIndex < 0 or len(ENVS) <= envIndex:
        print(f"Invalid env selected '{envIndex}'")
        return 1
    if algoIndex < 0 or len(ALGOS) <= algoIndex:
        print(f"Invalid algo selected '{algoIndex}'")
        return 1

    algoType = ALGOS[algoIndex]
    env = EnvWrapperFactory(ENVS[envIndex])
    env.Reset()

    qTableFile = f"{ENVS[envIndex]}_qtable.pkl"
    if algoType.lower() == "bruteforce":
        print("Bruteforcing it")
        print("Finished search")
        return 0
    elif algoType.lower() == "qlearning":
        print(f"Q-learning it\nParameters:\n  env={ENVS[envIndex]}\n  episodes={maxEpisodes}\n  epochs={maxEpochs}")
        print(f"\n  epsilon={EPSILON}\n  gamma={DISCOUNT_RATE}\n  alpha={LEARNING_RATE}\n")
        agent = QLearningAgent()
        agent.SetParameters(EPSILON, DISCOUNT_RATE, LEARNING_RATE, maxEpisodes, maxEpochs)
        policy = agent.CreatePolicyFunction()
        # load q-table if available
        if SHOULD_REUSE_QTABLE and os.path.exists(qTableFile):
            qTable = LoadFromPickle(qTableFile)
            print("Loaded q-table")
            resultsTrain = None
            resultsTest, globalRuntime = agent.Evaluate(env, qTable, verbose=verbose)
        else:
            resultsTrain, globalRuntime = agent.Train(env, policy, verbose=verbose)
            qTable = agent.QValues()
            SaveAsPickle(qTable, qTableFile)
            SaveAsPickle(resultsTrain, f"{algoType}_{ENVS[envIndex]}_train_{postfix}.pkl")
            print(f"Finished training: {globalRuntime: .4f}s")
            resultsTest, globalRuntime = agent.Evaluate(env, verbose=verbose)

        SaveAsPickle(resultsTest, f"{algoType}_{ENVS[envIndex]}_test_{postfix}.pkl")
        print(f"Finished evaluation: {globalRuntime: .4f}s")
    elif algoType.lower() == "a2c":
        # TODO: pipe in a2c
        print("nothing to see here...")
    elif algoType.lower() == "ddpg":
        discountRate = args.discountRate
        hiddenSize = args.hiddenLayerWidth
        batchSize = args.batchSize
        memoryRate = args.memoryRate
        alr = args.actorLearningRate
        clr = args.criticLearningRate
        print(f"DDPG\nParameters:\n  env={ENVS[envIndex]}\n  episodes={maxEpisodes}\n  epochs={maxEpochs}")
        print(f"  hiddenLayerSize={hiddenSize}\n  gamma={discountRate}\n  batchSize={batchSize}")
        ddpga = DdpgAgent(maxMemorySize=50000, maxEpisodes=maxEpisodes, maxEpochs=maxEpochs)
        resultsTrain, globalRuntime = ddpga.Train(env,
                                                  gamma=discountRate,
                                                  tau=memoryRate,
                                                  hiddenSize=hiddenSize,
                                                  actorLearningRate=alr,
                                                  criticLearningRate=clr,
                                                  batchSize=batchSize,
                                                  verbose=verbose)
        globalRewardAvg = np.mean([x.totalReward for x in resultsTrain])
        globalRewardStd = np.std([x.totalReward for x in resultsTrain])
        print(f"Finished training: {globalRuntime: .4f}s,\tavgEpisodeR = {globalRewardAvg: .4f} +/-{globalRewardStd: .4f}")
        SaveAsPickle(resultsTrain, f"{algoType}_{ENVS[envIndex]}_train_{postfix}.pkl")
        resultsTest, globalRuntime = ddpga.Test(env, verbose=verbose)
        globalRewardAvg = np.mean([x.totalReward for x in resultsTest])
        globalRewardStd = np.std([x.totalReward for x in resultsTest])
        print(f"Finished testing: {globalRuntime: .4f}s,\tavgEpisodeR = {globalRewardAvg: .4f} +/-{globalRewardStd: .4f}")
        SaveAsPickle(resultsTest, f"{algoType}_{ENVS[envIndex]}_test_{postfix}.pkl")
    else:
        print(f"Unsupported algo type '{algoType}'")
        return 1

    # Good practice to close env when finished :)
    env.Close()

    if not SHOULD_PLOT:
        return 0

    trainDescription = f"{algoType} {ENVS[envIndex]} training results {postfix}"
    testDescription = f"{algoType} {ENVS[envIndex]} test results {postfix}"

    figs = []
    figs.append(PlotPerformanceResults(resultsTrain, env.ActionSpaceLabels(shouldUseShorthand=True), trainDescription))
    figs.append(PlotPerformanceResults(resultsTest, env.ActionSpaceLabels(shouldUseShorthand=True), testDescription))

    for f in figs:
        SaveFigure(f)


if __name__ == "__main__":
    main()
