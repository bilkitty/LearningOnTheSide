import numpy as np
import random
from collections import defaultdict
from utils import RefreshScreen
from timeit import default_timer as timer

DEFAULT_ALPHA = 1
DEFAULT_EPSILON = 1
DEFAULT_GAMMA = 1
DEFAULT_MAX_EPOCHS = 100000
DEFAULT_MAX_EPISODES = 100000

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
        self.epsilon = DEFAULT_EPSILON
        self.gamma = DEFAULT_GAMMA
        self.alpha = DEFAULT_ALPHA
        self.maxEpisodes = DEFAULT_MAX_EPISODES
        self.maxEpochs = DEFAULT_MAX_EPOCHS

        self.qTable = None
        return

    def Train(self, env, policy, verbose=False):
        """
        inputs:
            env     obj           openai gym environment
            policy  func          a function that selects env actions in some way
        return:
            Metrics  performance results like timesteps, rewards, penalties, etc. per episode
            float    global training runtime
        """
        # Any newly seen state will be assigned q-values of zero for all states
        self.qTable = defaultdict(lambda: np.zeros(env.action_space.n))

        heading = f"In progress..."

        episodicMetrics = []
        globalStart = timer()
        for i in range(1, self.maxEpisodes + 1):
            epochs = totalReward = 0
            frames = []
            done = False
            s = env.reset()
            if isinstance(s, np.ndarray): s = tuple(s)

            start = timer()
            actionCounts = np.zeros(env.action_space.n)
            while not done:
                if random.uniform(0, 1) < self.epsilon:
                    a = env.action_space.sample()
                else:
                    a = np.argmax(self.qTable[s])  # Get maximizing parameter

                q = self.qTable[s][a]

                nextState, r, done, info = env.step(a)
                if isinstance(nextState, np.ndarray): nextState = tuple(nextState)

                nextHighestQ = np.max(self.qTable[nextState])  # Get maximal value

                # Update state and qtable
                self.qTable[s][a] = (1 - self.alpha) * q + self.alpha * (r + self.gamma * nextHighestQ)
                s = nextState

                actionCounts[a] += 1
                if verbose and epochs % (self.maxEpochs / 2) == 0:
                    RefreshScreen(mode="human")
                    print(f"{heading} \nTraining\ne={i}\nr={r}\nq={self.qTable[s][a]: .2f}")
                    totalCount = actionCounts.sum()
                    for b, cnt in enumerate(actionCounts): print(f"a{b}  {cnt / totalCount: .4f}")
                epochs += 1

            metrics = QLearningAgent.Metrics(frames, epochs, timer() - start, totalReward, done)
            metrics.SetActionCounts(actionCounts)
            episodicMetrics.append(metrics)

        return episodicMetrics, timer() - globalStart

    def Evaluate(self, env, policy):
        """
        inputs:
            env     obj           openai gym environment
            policy  func          a function that selects env actions in some way
        return:
            Metrics  performance results like timesteps, rewards, penalties, etc. per episode
            float    global test runtime
        """
        print("evaluate")
        assert self.qTable is not None
        return QLearningAgent.Metrics([], 0, 0.0, 0.0, False), 0.0

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

    def SetParameters(self,
                      epsilon=DEFAULT_EPSILON,
                      gamma=DEFAULT_GAMMA,
                      alpha=DEFAULT_ALPHA,
                      maxEpisodes=DEFAULT_MAX_EPISODES,
                      maxEpochs=DEFAULT_MAX_EPOCHS):
        """
        inputs:
            float   epsilon       probability of random action selection
            float   gamma         reward discount rate
            float   alpha         learning rate for q updates
            int     maxEpisodes   max number of episodes allowed
            int     maxEpochs     max number of time steps allowed
        return:
            n/a
        """
        self.epsilon = epsilon
        self.gamma = gamma
        self.alpha = alpha
        self.maxEpisodes = maxEpisodes
        self.maxEpochs = maxEpochs
        return

    def QValues(self):
        """
        inputs:
            n/a
        return:
            array   mxn q-value table for m states and n actions
        """
        print("qtable")
        assert self.qTable is not None
        return self.qTable



