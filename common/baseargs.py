import argparse
import utils
import os


class BaseArgs:
    def __init__(self, argsFromJson, cmdlnArgs):
        # Defaults
        self.maxEpisodes = 0
        self.maxEpochs = 0
        self.envIndex = 0
        self.algoIndex = 0

        # Overwrite defaults with json params
        for k, v in argsFromJson.items():
            setattr(self, k, v)

        # Commandline params take precedence
        self.maxEpochs = self.maxEpochs if cmdlnArgs.maxEpochs is None else cmdlnArgs.maxEpochs
        self.maxEpisodes = self.maxEpisodes if cmdlnArgs.maxEpisodes is None else cmdlnArgs.maxEpisodes
        self.envIndex = self.envIndex if cmdlnArgs.envIndex is None else cmdlnArgs.envIndex
        self.algoIndex = self.algoIndex if cmdlnArgs.algoIndex is None else cmdlnArgs.algoIndex
        self.verbose = cmdlnArgs.verbose


class BaseArgsParser:
    def __init__(self, description):
        self.cmdlnParser = argparse.ArgumentParser(description=description)
        self.cmdlnParser.add_argument("--maxEpochs", metavar="maxEpochs", type=int, help="")
        self.cmdlnParser.add_argument("--maxEpisodes", metavar="maxEpisodes", type=int, help="")
        self.cmdlnParser.add_argument("--envIndex", metavar="envIndex", type=int, help="")
        self.cmdlnParser.add_argument("--algoIndex", metavar="algoIndex", type=int, help="")
        self.cmdlnParser.add_argument("--tau", metavar="tau", type=float, help="")
        self.cmdlnParser.add_argument("--verbose", action="store_true", help="")

    def ParseArgs(self, jsonfilepath=None, argsList=None):
        jsonargs = {} if jsonfilepath is None else self.GetJsonArgs(jsonfilepath)
        cmdlnargs = self.cmdlnParser.parse_args() if argsList is None else self.cmdlnParser.parse_args(argsList)
        return BaseArgs(jsonargs, cmdlnargs)

    def GetJsonArgs(self, jsonfilepath):
        # TODO: validate ranges
        assert os.path.exists(jsonfilepath)
        return utils.LoadFromJsonFile(jsonfilepath)
