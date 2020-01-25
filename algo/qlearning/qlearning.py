import numpy as np
import random
from collections import defaultdict
from timeit import default_timer as timer
from utils import *
from metrics import *
from memory import Memory

DEFAULT_ALPHA = 0.6
DEFAULT_EPSILON = 0.1
DEFAULT_GAMMA = 0.1
DEFAULT_MAX_EPOCHS = GetMaxFloat()
DEFAULT_MAX_EPISODES = 100000


"""
TODO: Desc

Note that the qtable is a dictionary that provides default values states not found in table.
"""


class QTable:
    def __init__(self, envWrapper):
        self._table = defaultdict(lambda: np.zeros(envWrapper.ActionSpaceN()))
        self.numActions = envWrapper.ActionSpaceN()

    def GetValue(self, state, action):
        assert(isinstance(action, np.int32) or isinstance(action, np.int64))
        if isinstance(state, np.ndarray): state = tuple(state)
        return self._table[state][action]

    def GetMaxValue(self, state):
        if isinstance(state, np.ndarray): state = tuple(state)
        return np.max(self._table[state])

    def GetArgMax(self, state):
        if isinstance(state, np.ndarray): state = tuple(state)
        return np.argmax(self._table[state])

    def SetValue(self, state, action, value):
        assert(isinstance(action, np.int32) or isinstance(action, np.int64))
        if isinstance(state, np.ndarray): state = tuple(state)
        self._table[state][action] = value

    def GetTable(self):
        return dict(self._table)


class QLearningAgent:

    def __init__(self,
                 epsilon=DEFAULT_EPSILON,
                 gamma=DEFAULT_GAMMA,
                 alpha=DEFAULT_ALPHA,
                 maxEpisodes=DEFAULT_MAX_EPISODES,
                 maxEpochs=DEFAULT_MAX_EPOCHS,
                 batchSize=1):
        """
        args:
            float       epsilon       probability of random action selection
            float       gamma         reward discount rate
            float       alpha         learning rate for q updates
            int         maxEpisodes   max number of episodes allowed
            int         maxEpochs     max number of time steps allowed
            int         batchSize     number of samples to collection from past experience
        return:
            n/a
        """
        self.epsilon = epsilon
        self.gamma = gamma
        self.alpha = alpha
        self.maxEpisodes = maxEpisodes
        self.maxEpochs = maxEpochs
        self.batchSize = batchSize

        self.qTable = None
        self.experiences = None
        self.policy = None
        return

    # TODO: get rid of train and test; supplant with getaction(), update(),...

    def Initialize(self, envWrapper, maxMemorySize, qTable=None):
        """
        args:
            obj      envWrapper    wrapper containing gym env
            int      maxMemorySize size limit for experience buffer
        return:
            n/a
        """
        self.experiences = Memory(maxMemorySize)
        self.qTable = QTable(envWrapper)
        if qTable is None: qTable = self.qTable
        self.policy = QLearningAgent.CreatePolicyFunction(qTable, self.epsilon)

    def Update(self):
        # TODO: make compatible with variable batch sizes
        state, action, reward, nextState, _ = self.experiences.getLatest()

        nextHighestQ = self.qTable.GetMaxValue(nextState)
        q = self.qTable.GetValue(state, action)
        newQ = (1 - self.alpha) * q + self.alpha * (reward + self.gamma * nextHighestQ)
        self.qTable.SetValue(state, action, newQ)

    def SaveExperience(self, state, action, nextState, reward, done):
        # TODO: [expmt] try spacing these out?
        self.experiences.push(state=state, action=action, reward=reward, next_state=nextState, done=done)

    def GetBestAction(self, state):
        return self.qTable.GetArgMax(state)

    def GetAction(self, state):
        return self.policy(state)

    def GetValue(self, state, action):
        return self.qTable.GetValue(state, action)

    def SaveLearntModel(self, filepath):
        valsToSave = self.qTable.GetTable()
        raise NotImplementedError

    def LoadLearntModel(self, filepath):
        raise NotImplementedError

    @staticmethod
    def CreatePolicyFunction(qt, epsilon):
        """
        args:
            QTable  qt          an LUT of action-values per state
        return:
                    func        function that generates action based on state
        """
        def EpsilonGreedyPolicy(s):
            if random.uniform(0, 1) < epsilon:
                a = np.random.choice(np.arange(qt.numActions))
            else:
                a = qt.GetArgMax(s)

            return a

        return EpsilonGreedyPolicy

    @staticmethod
    def PlotValues():
        """
        args:
            envType ?
            qTable  dict
        return:
                    figure  figure containing plots of q-values for particular states
        """

        # TODO: how to represent states generically? What aspects in the env are useful for
        #       deciding how to interpret state info?

    def Train(self, env, policy, verbose=False):
        """
        args:
            env          obj           openai gym environment
            policy       func          a function that selects env actions in some way
            verbose      bool          (optional) enables console output when True
        return:
                         Metrics       performance results like timesteps, rewards, penalties, etc. per episode
                         float         global training runtime
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
        args:
            env          obj           openai gym environment
            qTable       dict          (optional) an LUT for best actions for every state
            verbose      bool          (optional) enables console output when True
        return:
                         Metrics       performance results like timesteps, rewards, penalties, etc. per episode
                         float         global test runtime
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

