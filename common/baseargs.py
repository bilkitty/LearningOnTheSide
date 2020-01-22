import argparse
import utils
import os


class BaseArgs:
    def __init__(self, fromjson, fromcmdln):
        for k, v in fromjson.items():
            setattr(self, k, v)

        #for i, e in enumerate(fromcmdln):
        #    setattr(self, i, e)


class BaseArgsParser:
    def __init__(self, description):
        self.cmdlnParser = argparse.ArgumentParser(description=description)
        self.cmdlnParser.add_argument("maxEpochs", metavar="maxEpochs", type=int, help="")
        self.cmdlnParser.add_argument("maxEpisodes", metavar="maxEpisodes", type=int, help="")
        self.cmdlnParser.add_argument("envIndex", metavar="envIndex", type=int, help="")
        self.cmdlnParser.add_argument("algoIndex", metavar="algoIndex", type=int, help="")

    def ParseArgs(self, jsonfilepath=None):
        cmdlnargs_whatisthis = self.cmdlnParser.parse_args()
        jsonargs = {} if jsonfilepath is None else self.GetJsonArgs(jsonfilepath)
        return BaseArgs(self.GetJsonArgs(jsonfilepath), cmdlnargs_whatisthis)

    def GetJsonArgs(self, jsonfilepath):
        # TODO: validate ranges
        assert os.path.exists(jsonfilepath)
        return utils.LoadFromJsonFile(jsonfilepath)
