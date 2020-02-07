import sys, os
import numpy as np

from utils import *
from rollouts import *
from args import *
from baseargs import BaseArgsParser
from actorcritic.ddpg import DdpgAgent
from qlearning.qlearning import QLearningAgent
from environments import EnvTypes, EnvWrapperFactory
from visualise import PlotPerformanceResults, SaveFigure

MAX_MEMORY_SIZE = 50000
IS_ENV_DETERMINISTIC = False
PARAM_FILE = os.path.join(GetRootProjectPath(), "gymrunner/params.json")

ENVS = [EnvTypes.WindyGridEnv, EnvTypes.TaxiGridEnv, EnvTypes.CartPoleEnv,
        EnvTypes.AcroBotEnv, EnvTypes.MountainCarEnv, EnvTypes.ContinuousPendulumEnv,
        EnvTypes.ContinuousMountainCarEnv]

ALGOS = ["qlearning", "ddpg"]


def main():
    baseArgs = BaseArgsParser().ParseArgs(PARAM_FILE)
    envIndex = baseArgs.envIndex
    algoIndex = baseArgs.algoIndex
    maxEpisodes = baseArgs.maxEpisodes
    maxEpochs = baseArgs.maxEpochs
    verbose = baseArgs.verbose
    shouldTestOnly = baseArgs.shouldTestOnly
    shouldSkipPlots = baseArgs.shouldSkipPlots
    postfix = baseArgs.desc

    if envIndex < 0 or len(ENVS) <= envIndex:
        print(f"Invalid env selected '{envIndex}'")
        return 1
    if algoIndex < 0 or len(ALGOS) <= algoIndex:
        print(f"Invalid algo selected '{algoIndex}'")
        return 1

    header = ""
    if verbose:
        header += "(verbose mode)\n"

    agent = None
    algoType = ALGOS[algoIndex]
    env = EnvWrapperFactory(ENVS[envIndex], isDeterministic=IS_ENV_DETERMINISTIC)
    env.Reset()

    if algoType.lower() == "bruteforce":
        print("Bruteforcing it")
        print("Finished search")
        return 0

    elif algoType.lower() == "qlearning":
        args = QLearningArgsParser().ParseArgs(PARAM_FILE)
        header += f"Q-learning\nParameters:\n  env={ENVS[envIndex]}\n  episodes={maxEpisodes}\n  epochs={maxEpochs}\n"
        header += f"  epsilon={args.exploreRate}\n  gamma={args.discountRate}\n  alpha={args.learnRate}\n"
        agent = QLearningAgent(maxEpisodes=maxEpisodes,
                               maxEpochs=maxEpochs,
                               epsilon=args.exploreRate,
                               gamma=args.discountRate,
                               alpha=args.learnRate,
                               batchSize=args.batchSize)

        # TODO: choose binning based on env characteristics; either query binning from env, or implement it within.
        #       can't just set a fixed number because each dim of state/action may have very different val ranges.
        #       Perhaps at this level, we can specify coarse-to-fine ranges which will be implemented within env?
        mActionBins = np.array([2] * env.ActionSpaceN())
        nStateBins = np.array([100, 30, 100, 30])
        agent.Initialize(env, MAX_MEMORY_SIZE, mActionBins, nStateBins)

    elif algoType.lower() == "a2c":
        # TODO: pipe in a2c
        header += "nothing to see here...\n"

    elif algoType.lower() == "ddpg":
        args = DdpgArgsParser().ParseArgs(PARAM_FILE)
        hiddenLayerWidth = np.array([args.hiddenLayerWidth0, args.hiddenLayerWidth1])
        header += f"DDPG\nParameters:\n  env={ENVS[envIndex]}\n  episodes={maxEpisodes}\n  epochs={maxEpochs}\n"
        header += f"  hiddenLayerSize={hiddenLayerWidth}\n  gamma={args.discountRate}\n  batchSize={args.batchSize}\n"
        header += f"  softUpdate={args.softUpdateRate}\n  noiseShift={args.noiseShift}\n  noiseScale={args.noiseScale}\n"
        agent = DdpgAgent(maxEpisodes=maxEpisodes,
                          maxEpochs=maxEpochs,
                          gamma=args.discountRate,
                          tau=args.softUpdateRate,
                          batchSize=args.batchSize)

        agent.Initialize(env,
                         maxMemorySize=MAX_MEMORY_SIZE,
                         hiddenSize=hiddenLayerWidth,
                         actorLearningRate=args.actorLearnRate,
                         criticLearningRate=args.criticLearnRate,
                         noiseShift=args.noiseShift,
                         noiseScale=args.noiseScale)

    else:
        print(f"Unsupported algo type '{algoType}'")
        return 1

    if not shouldTestOnly:
        resultsTrain, globalRuntime = Train(env, agent, verbose=verbose, headerText=header)
        # TODO: save learnt model
        globalRewardAvg = np.mean([x.totalReward for x in resultsTrain])
        globalRewardStd = np.std([x.totalReward for x in resultsTrain])
        print(f"Finished training: {globalRuntime: .4f}s,\tavgEpisodeR = {globalRewardAvg: .4f} +/-{globalRewardStd: .4f}")
        SaveAsPickle(resultsTrain, f"{algoType}_{ENVS[envIndex]}_train_{postfix}.pkl")
    else:
        # TODO: load learnt model
        print("Loading model...")

    resultsTest, globalRuntime = Test(env, agent, verbose=verbose, headerText=header)
    globalRewardAvg = np.mean([x.totalReward for x in resultsTest])
    globalRewardStd = np.std([x.totalReward for x in resultsTest])
    print(f"Finished testing: {globalRuntime: .4f}s,\tavgEpisodeR = {globalRewardAvg: .4f} +/-{globalRewardStd: .4f}")
    SaveAsPickle(resultsTest, f"{algoType}_{ENVS[envIndex]}_test_{postfix}.pkl")
    # Good practice to close env when finished :)
    env.Close()

    if shouldSkipPlots:
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
