import argparse
import utils
import os


class BaseArgs:
    DEFAULT_MAX_EPISODES = 1
    DEFAULT_MAX_EPOCHS = 1
    DEFAULT_ENV_INDEX = 0
    DEFAULT_ALGO_INDEX = 0
    DEFAULT_DESCRIPTION = ""

    def __init__(self, argsFromJson, cmdlnArgs):
        # Defaults for non-boolean params
        self.maxEpochs = BaseArgs.DEFAULT_MAX_EPOCHS
        self.maxEpisodes = BaseArgs.DEFAULT_MAX_EPISODES
        self.envIndex = BaseArgs.DEFAULT_ENV_INDEX
        self.algoIndex = BaseArgs.DEFAULT_ALGO_INDEX
        self.desc = BaseArgs.DEFAULT_DESCRIPTION

        # Overwrite defaults with json params
        for k, v in argsFromJson.items():
            setattr(self, k, v)

        # Commandline params take precedence
        self.maxEpochs = self.maxEpochs if cmdlnArgs.maxEpochs is None else cmdlnArgs.maxEpochs
        self.maxEpisodes = self.maxEpisodes if cmdlnArgs.maxEpisodes is None else cmdlnArgs.maxEpisodes
        self.envIndex = self.envIndex if cmdlnArgs.envIndex is None else cmdlnArgs.envIndex
        self.algoIndex = self.algoIndex if cmdlnArgs.algoIndex is None else cmdlnArgs.algoIndex
        self.verbose = cmdlnArgs.verbose
        self.shouldTestOnly = cmdlnArgs.testonly
        self.shouldSkipPlots = cmdlnArgs.noplots


class BaseArgsParser:
    def __init__(self):
        eidHelpMsg = """{\n 0 : WindyGridEnv, \n 1 : TaxiGridEnv, \n 2 : CartPoleEnv,
        \n 3 : AcroBotEnv, \n 4 : MountainCarEnv, \n 5 : ContinuousPendulumEnv,
        \n 6 : ContinuousMountainCarEnv\n}"""
        aidHelpMsg = """{\n 0 : qlearning, \n 1 : ddpg\n}"""

        self.cmdlnParser = argparse.ArgumentParser(description="")
        self.cmdlnParser.add_argument("--maxEpochs", metavar="epochs", type=int, help="Max time steps per rollout")
        self.cmdlnParser.add_argument("--maxEpisodes", metavar="rollouts", type=int, help="Max rollouts")
        self.cmdlnParser.add_argument("--envIndex", metavar="envId", type=int, help=eidHelpMsg)
        self.cmdlnParser.add_argument("--algoIndex", metavar="algoId", type=int, help=aidHelpMsg)
        self.cmdlnParser.add_argument("--verbose", action="store_true", help="Display additional info")
        self.cmdlnParser.add_argument("--testonly", action="store_true", help="Load learnt models and test")
        self.cmdlnParser.add_argument("--noplots", action="store_true", help="Skip plotting")

    def ParseArgs(self, jsonfilepath=None, argsList=None):
        jsonargs = {} if jsonfilepath is None else utils.LoadFromJsonFile(jsonfilepath)
        cmdlnargs = self.cmdlnParser.parse_args() if argsList is None else self.cmdlnParser.parse_args(argsList)
        return BaseArgs(jsonargs, cmdlnargs)


