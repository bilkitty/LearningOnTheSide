import pickle as pkl


class Metrics:

    def __init__(self, frames, epochs, globalRuntime, totalReward, success):
        self.frames = frames
        self.epochCount = epochs
        self.globalRuntime = globalRuntime
        self.totalReward = totalReward
        self.success = success
        self.actionCounts = []

    def SetActionCounts(self, actionCounts):
        self.actionCounts = actionCounts

    def Str(self):
        s = ""
        for frame in self.frames[len(self.frames) - 5:]:
            s += frame["frame"]
            s += "\n"
        return f"{s}\ne: {self.epochCount}\np: {self.globalRuntime}\nr: {self.totalReward}\nsuccess: {self.success}\n"

    @staticmethod
    def LoadMetricsFromPickle(filepath):
        f = open(filepath, "rb")
        metrics = pkl.load(f)
        return metrics

    @staticmethod
    def SaveMetricsAsPicle(metrics, filepath):
        f = open(filepath, "wb")
        pkl.dump(metrics, f)
        f.close()