import numpy as np


class QLearningAgent:

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

    def __init__(self):
        """
        inputs:
            n/a
        return:
            n/a
        """
        return

    def Train(self, env, policy):
        """
        inputs:
            env     obj           openai gym environment
            policy  func          a function that selects env actions in some way
        return:
            bool    true for successful
            Metrics performance results like timesteps, rewards, penalties, etc. per episode
        """
        print("train")
        return False, QLearningAgent.Metrics([], 0, 0.0, 0.0, False)

    def Evaluate(self, env, policy):
        """
        inputs:
            env     obj           openai gym environment
            policy  func          a function that selects env actions in some way
        return:
            bool    true for successful
            Metrics performance results like timesteps, rewards, penalties, etc. per episode
        """
        print("evaluate")
        return False, QLearningAgent.Metrics([], 0, 0.0, 0.0, False)

    def CreatePolicy(self):
        """
        inputs:
            obj     state         current state representation
            float   epsilon       probability of choosing random vs best action
        return:
            func    function that generates action based on state
        """
        print("policy creation")
        return lambda x: x + 1

    def SetParameters(self):
        """
        inputs:
            int     maxEpochs     max number of time steps allowed
            int     maxEpisodes   max number of episodes allowed
            float   epsilon       probability of random action selection
            float   gamma         reward discount rate
            float   alpha         learning rate for q updates
        return:
            n/a
        """
        print("params")
        return

    def QValues(self):
        """
        inputs:
            n/a
        return:
            array   mxn q-value table for m states and n actions
        """
        print("qtable")
        arr = []
        return arr



