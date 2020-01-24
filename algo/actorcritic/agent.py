DEFAULT_MAX_EPISODES = 100000
DEFAULT_MAX_EPOCHS = 100000


class ModelFreeAgent:
    def __init__(self, maxEpisodes=DEFAULT_MAX_EPISODES, maxEpochs=DEFAULT_MAX_EPOCHS):
        self.maxEpisodes = maxEpisodes
        self.maxEpochs = maxEpochs

    def Update(self, *args, **kwargs):
        raise NotImplementedError

    def SaveExperience(self, state, action, nextState, reward, done):
        raise NotImplementedError

    # TODO: should use **kwargs? What does that usually mean?
    def GetAction(self, state, shouldAddNoise=True):
        raise NotImplementedError

    def GetValue(self, state, action):
        raise NotImplementedError
