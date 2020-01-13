import numpy as np
import random
from collections import defaultdict
from timeit import default_timer as timer
from matplotlib import pyplot as plt
from utils import RefreshScreen

DEFAULT_ALPHA = 1
DEFAULT_EPSILON = 1
DEFAULT_GAMMA = 1
DEFAULT_MAX_EPOCHS = float("inf")
DEFAULT_MAX_EPISODES = 100000
RENDERING_MODE="human"

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
            env      obj           openai gym environment
            policy   func          a function that selects env actions in some way
            verbose  bool          (optional) enables console output when True
        return:
            Metrics  performance results like timesteps, rewards, penalties, etc. per episode
            float    global training runtime
        """
        # Any newly seen state will be assigned q-values of zero for all states
        self.qTable = defaultdict(lambda: np.zeros(env.action_space.n))

        heading = f"In progress..."

        episodicMetrics = []
        globalStart = timer()
        try:
            for i in range(1, self.maxEpisodes + 1):
                epochs = totalReward = 0
                frames = []
                done = False
                s = env.reset()
                if isinstance(s, np.ndarray): s = tuple(s)

                start = timer()
                actionCounts = np.zeros(env.action_space.n)
                while not done and epochs < self.maxEpochs:
                    a = policy(s)

                    q = self.qTable[s][a]

                    nextState, r, done, info = env.step(a)
                    if isinstance(nextState, np.ndarray): nextState = tuple(nextState)

                    # Update state and q-value
                    nextHighestQ = np.max(self.qTable[nextState])
                    self.qTable[s][a] = (1 - self.alpha) * q + self.alpha * (r + self.gamma * nextHighestQ)
                    s = nextState

                    actionCounts[a] += 1
                    if verbose and epochs % (self.maxEpochs / 2) == 0:
                        RefreshScreen(mode=RENDERING_MODE)
                        print(f"{heading} \nTraining\ne={i}\nr={r}\nq={self.qTable[s][a]: .2f}")
                        totalCount = actionCounts.sum()
                        for b, cnt in enumerate(actionCounts): print(f"a{b}  {cnt / totalCount: .4f}")

                        # yeh, yeh, skipped the first state
                        frames.append({
                            'frame': env.render(mode=RENDERING_MODE),
                            'state': s,
                            'action': a,
                            'reward': r})

                    epochs += 1

                metrics = QLearningAgent.Metrics(frames, epochs, timer() - start, totalReward, done)
                metrics.SetActionCounts(actionCounts)
                episodicMetrics.append(metrics)

        except MemoryError:
            env.close()

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
            qTable = defaultdict(lambda: np.zeros(env.action_space.n))

        heading = f"In progress..."
        episodicMetrics = []
        globalStart = timer()
        try:
            for i in range(1, self.maxEpisodes + 1):
                epochs = totalReward = 0
                frames = []
                done = False
                s = env.reset()
                if isinstance(s, np.ndarray): s = tuple(s)

                start = timer()
                actionCounts = np.zeros(env.action_space.n)
                while not done and epochs < self.maxEpochs:
                    # Always exploit
                    a = np.argmax(qTable[s])

                    s, r, done, info = env.step(a)
                    if isinstance(s, np.ndarray): s = tuple(s)

                    actionCounts[a] += 1
                    if verbose and epochs % (self.maxEpochs / 2) == 0:
                        RefreshScreen(mode=RENDERING_MODE)
                        print(f"{heading} \nTesting\ne={i}\nr={r}\nq={qTable[s][a]: .2f}")
                        totalCount = actionCounts.sum()
                        for b, cnt in enumerate(actionCounts): print(f"a{b}  {cnt / totalCount: .4f}")

                        # yeh, yeh, skipped the first state
                        frames.append({
                            'frame': env.render(mode=RENDERING_MODE),
                            'state': s,
                            'action': a,
                            'reward': r})

                    epochs += 1

                metrics = QLearningAgent.Metrics(frames, epochs, timer() - start, totalReward, done)
                metrics.SetActionCounts(actionCounts)
                episodicMetrics.append(metrics)

        except MemoryError:
            env.close()

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

    def PlotQValuesForEnv(self):
        """
        inputs:
            envType ?
        return:
            obj     figure containing plots of q-values for particular states
        """

