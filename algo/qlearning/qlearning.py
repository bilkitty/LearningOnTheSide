import numpy as np
import random
from collections import defaultdict
from timeit import default_timer as timer
from utils import *
from metrics import *

DEFAULT_ALPHA = 0.6
DEFAULT_EPSILON = 0.1
DEFAULT_GAMMA = 0.1
DEFAULT_MAX_EPOCHS = GetMaxFloat()
DEFAULT_MAX_EPISODES = 100000


"""
TODO: Desc
"""


class QLearningAgent:

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
            env      obj           openai gym environment
            policy   func          a function that selects env actions in some way
            verbose  bool          (optional) enables console output when True
        return:
                     Metrics  performance results like timesteps, rewards, penalties, etc. per episode
                     float    global training runtime
        """
        # Any newly seen state will be assigned q-values of zero for all states
        self.qTable = defaultdict(lambda: np.zeros(env.ActionSpaceN()))

        heading = f"In progress..."

        episodicMetrics = []
        globalStart = timer()
        try:
            for i in range(1, self.maxEpisodes + 1):
                epochs = totalReward = 0
                frames = []
                done = False
                s = env.Reset()
                if isinstance(s, np.ndarray): s = tuple(s)

                start = timer()
                actionCounts = np.zeros(env.ActionSpaceN())
                while not done and epochs < self.maxEpochs:
                    a = policy(s)

                    q = self.qTable[s][a]

                    nextState, r, done, info = env.Step(a)
                    if isinstance(nextState, np.ndarray): nextState = tuple(nextState)

                    # Update state and q-value
                    nextHighestQ = np.max(self.qTable[nextState])
                    self.qTable[s][a] = (1 - self.alpha) * q + self.alpha * (r + self.gamma * nextHighestQ)
                    s = nextState

                    actionCounts[a] += 1
                    if verbose and epochs % (self.maxEpochs / 2) == 0:
                        RefreshScreen(mode="human")
                        print(f"{heading} \nTraining\ne={i}\nr={r}\nq={self.qTable[s][a]: .2f}")
                        totalCount = actionCounts.sum()
                        for b, cnt in enumerate(actionCounts): print(f"a{b}  {cnt / totalCount: .4f}")

                        # yeh, yeh, skipped the first state
                        frames.append({
                            'frame': env.Render(),
                            'state': s,
                            'action': a,
                            'reward': r})

                    epochs += 1
                    totalReward += r

                metrics = Metrics(frames, epochs, timer() - start, totalReward, done)
                metrics.SetActionCounts(actionCounts)
                episodicMetrics.append(metrics)

        except MemoryError:
            env.Close()

        return episodicMetrics, timer() - globalStart

    def Evaluate(self, env, qTable=None, verbose=False):
        """
        inputs:
            env      obj           openai gym environment
            qTable   dict          (optional) an LUT for best actions for every state
            verbose  bool          (optional) enables console output when True
        return:
                     Metrics  performance results like timesteps, rewards, penalties, etc. per episode
                     float    global test runtime
        """
        if qTable is None:
            assert self.qTable is not None
            qTable = self.qTable
        else:
            # A q-table was likely loaded from disk
            assert isinstance(qTable, dict)
            qTable = defaultdict(lambda: np.zeros(env.ActionSpaceN()))

        heading = f"In progress..."
        episodicMetrics = []
        globalStart = timer()
        try:
            for i in range(1, self.maxEpisodes + 1):
                epochs = totalReward = 0
                frames = []
                done = False
                s = env.Reset()
                if isinstance(s, np.ndarray): s = tuple(s)

                start = timer()
                actionCounts = np.zeros(env.ActionSpaceN())
                while not done and epochs < self.maxEpochs:
                    # Always exploit
                    a = np.argmax(qTable[s])

                    s, r, done, info = env.Step(a)
                    if isinstance(s, np.ndarray): s = tuple(s)

                    actionCounts[a] += 1
                    if verbose and epochs % (self.maxEpochs / 2) == 0:
                        RefreshScreen(mode="human")
                        print(f"{heading} \nTesting\ne={i}\nr={r}\nq={qTable[s][a]: .2f}")
                        totalCount = actionCounts.sum()
                        for b, cnt in enumerate(actionCounts): print(f"a{b}  {cnt / totalCount: .4f}")

                        # yeh, yeh, skipped the first state
                        frames.append({
                            'frame': env.Render(),
                            'state': s,
                            'action': a,
                            'reward': r})

                    epochs += 1
                    totalReward += r

                metrics = Metrics(frames, epochs, timer() - start, totalReward, done)
                metrics.SetActionCounts(actionCounts)
                episodicMetrics.append(metrics)

        except MemoryError:
            env.Close()

        return episodicMetrics, timer() - globalStart

    def CreatePolicyFunction(self, qTable=None):
        """
        inputs:
            dict    qTable        an LUT of action-values per state
        return:
                    func    function that generates action based on state
        """
        def EpsilonGreedyPolicy(s):
            assert(self.qTable is not None or qTable is not None)
            actions = self.qTable[s] if qTable is None else qTable[s]

            if random.uniform(0, 1) < self.epsilon:
                a = np.random.choice(np.arange(len(actions)))
            else:
                a = np.argmax(actions)

            return a

        return EpsilonGreedyPolicy

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
                    dict    contains the rows of virtual mxn q-value table for m states and n actions
        """
        assert self.qTable is not None
        return dict(self.qTable)

    @staticmethod
    def PlotActionValues():
        """
        inputs:
            envType ?
            qTable  dict
        return:
                    figure  figure containing plots of q-values for particular states
        """

        # TODO: how to represent states generically? What aspects in the env are useful for
        #       deciding how to interpret state info?

