import argparse
import utils
import os


class BaseArgs:
    def __init__(self, argsFromJson, cmdlnArgs):
        for k, v in argsFromJson.items():
            setattr(self, k, v)

        # TODO: check that attribute exists in jsonArgs
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

    def ParseArgs(self, jsonfilepath=None):
        jsonargs = {} if jsonfilepath is None else self.GetJsonArgs(jsonfilepath)
        return BaseArgs(jsonargs, self.cmdlnParser.parse_args())

    def GetJsonArgs(self, jsonfilepath):
        # TODO: validate ranges
        assert os.path.exists(jsonfilepath)
        return utils.LoadFromJsonFile(jsonfilepath)
