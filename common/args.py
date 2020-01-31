from baseargs import *


class DdpgArgs(BaseArgs):
    DEFAULT_REWARD_DISCOUNT_RATE = 0.1
    DEFAULT_TARGET_UPDATE_RATE = 0
    DEFAULT_HIDDEN_WIDTH = 16
    DEFAULT_BATCH_SIZE = 128
    DEFAULT_UPDATE_RATE = 0.01
    DEFAULT_ACTOR_LR = 1e-4
    DEFAULT_CRITIC_LR = 1e-4
    DEFAULT_NOISE_SHIFT = 0.3
    DEFAULT_NOISE_SCALE = 0.15

    def __init__(self, argsFromJson, cmdlnArgs):
        BaseArgs.__init__(self, argsFromJson, cmdlnArgs)
        # Defaults for non-boolean params
        self.softUpdateRate = DdpgArgs.DEFAULT_TARGET_UPDATE_RATE
        self.discountRate = DdpgArgs.DEFAULT_REWARD_DISCOUNT_RATE
        self.batchSize = DdpgArgs.DEFAULT_BATCH_SIZE
        self.hiddenLayerWidth0 = DdpgArgs.DEFAULT_HIDDEN_WIDTH
        self.hiddenLayerWidth1 = DdpgArgs.DEFAULT_HIDDEN_WIDTH
        self.actorLearnRate = DdpgArgs.DEFAULT_ACTOR_LR
        self.criticLearnRate = DdpgArgs.DEFAULT_CRITIC_LR
        self.noiseShift = DdpgArgs.DEFAULT_NOISE_SHIFT
        self.noiseScale = DdpgArgs.DEFAULT_NOISE_SCALE

        # Overwrite defaults with json params
        for k, v in argsFromJson.items():
            setattr(self, k, v)

        # Commandline params take precedence
        self.softUpdateRate = self.softUpdateRate if cmdlnArgs.softUpdateRate is None else cmdlnArgs.softUpdateRate
        self.discountRate = self.discountRate if cmdlnArgs.discountRate is None else cmdlnArgs.discountRate
        self.noiseShift = self.noiseShift if cmdlnArgs.noiseShift is None else cmdlnArgs.noiseShift
        self.noiseScale = self.noiseScale if cmdlnArgs.noiseScale is None else cmdlnArgs.noiseScale


class DdpgArgsParser(BaseArgsParser):
    def __init__(self):
        BaseArgsParser.__init__(self)
        self.cmdlnParser.add_argument("--softUpdateRate", metavar="target", type=float, help="For target networks")
        self.cmdlnParser.add_argument("--discountRate", metavar="discount", type=float, help="Reward discount rate")
        self.cmdlnParser.add_argument("--noiseShift", metavar="sigmaNoise", type=float, help="Noise shift")
        self.cmdlnParser.add_argument("--noiseScale", metavar="thetaNoise", type=float, help="Noise scale")

    def ParseArgs(self, jsonfilepath=None, argsList=None):
        jsonargs = {} if jsonfilepath is None else utils.LoadFromJsonFile(jsonfilepath)
        cmdlnargs = self.cmdlnParser.parse_args() if argsList is None else self.cmdlnParser.parse_args(argsList)
        return DdpgArgs(jsonargs, cmdlnargs)

    # TODO: validate ranges


class QLearningArgs(BaseArgs):
    DEFAULT_RANDOM_ACTION_PROB_THRESH = 0
    DEFAULT_LEARNING_RATE = 0.6
    DEFAULT_REWARD_DISCOUNT_RATE = 0.1
    DEFAULT_BATCH_SIZE = 128

    def __init__(self, argsFromJson, cmdlnArgs):
        BaseArgs.__init__(self, argsFromJson, cmdlnArgs)
        # Defaults for non-boolean params
        self.exploreRate = QLearningArgs.DEFAULT_RANDOM_ACTION_PROB_THRESH
        self.learnRate = QLearningArgs.DEFAULT_LEARNING_RATE
        self.discountRate = QLearningArgs.DEFAULT_REWARD_DISCOUNT_RATE
        self.batchSize = QLearningArgs.DEFAULT_BATCH_SIZE

        # Overwrite defaults with json params
        for k, v in argsFromJson.items():
            setattr(self, k, v)

        # Commandline params take precedence
        self.discountRate = self.discountRate if cmdlnArgs.discountRate is None else cmdlnArgs.discountRate
        self.learnRate = self.learnRate if cmdlnArgs.learnRate is None else cmdlnArgs.learnRate
        self.exploreRate = self.exploreRate if cmdlnArgs.exploreRate is None else cmdlnArgs.exploreRate


class QLearningArgsParser(BaseArgsParser):
    def __init__(self):
        BaseArgsParser.__init__(self)
        self.cmdlnParser.add_argument("--discountRate", metavar="discount", type=float, help="Reward discount rate")
        self.cmdlnParser.add_argument("--learnRate", metavar="lr", type=float, help="Learning rate during updates")
        self.cmdlnParser.add_argument("--exploreRate", metavar="explore", type=float, help="Min prob of random action")

    def ParseArgs(self, jsonfilepath=None, argsList=None):
        jsonargs = {} if jsonfilepath is None else utils.LoadFromJsonFile(jsonfilepath)
        cmdlnargs = self.cmdlnParser.parse_args() if argsList is None else self.cmdlnParser.parse_args(argsList)
        return QLearningArgs(jsonargs, cmdlnargs)

    # TODO: validate ranges
