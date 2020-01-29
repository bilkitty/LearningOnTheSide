from torch.autograd import Variable
from timeit import default_timer as timer

from memory import *
from metrics import *
from .nnModels import *
from .agent import ModelFreeAgent
from utils import RefreshScreen
from noiseprocess import OUStrategy

# TODO: this version of ddpg only supports continuous envs. The interpretation of action space needs to be compatible
#       with Discrete and Box types. -_-


class DdpgAgent(ModelFreeAgent):

    def __init__(self, maxEpisodes, maxEpochs, gamma, tau, batchSize):
        """
        args:
            int         maxEpisodes     limit of rollouts
            int         maxEpochs       limit of time steps in rollouts
            float       gamma           reward discount rate
            float       tau             soft update rate for target networks
            int         bSize           sample size for updates using replay

        """

        ModelFreeAgent.__init__(self, maxEpisodes=maxEpisodes, maxEpochs=maxEpochs)
        self.gamma = min(max(0, gamma), 1)
        self.tau = min(max(0, tau), 1)
        self.batchSize = batchSize

        self.actor = None
        self.actorTarget = None
        self.critic = None
        self.criticTarget = None
        self.actorOptimizer = None
        self.criticOptimizer = None
        self.noiseProcess = None
        self.experiences = None
        self.lossFunction = nn.MSELoss()

    def Initialize(self, envWrapper, maxMemorySize, hiddenSize, actorLearningRate, criticLearningRate, noiseShift, noiseScale):
        """
        args:
            obj         envWrapper      wrapper containing gym env
            int         hiddenSize      width of all hidden layers
            float       alRate          learning rate for actor optimization
            float       clRate          learning rate for critic optimization


            In the future, it would be awesome to create an agent that is less dependent on the training/test
            environment.
                The arch of the neural networks used by this agent is determined by action and state space size
                Similarly for the noise process

            So, how can these internal components be made more generic? Map specific action/state spaces to a
            more general interface? What are possible ways to "normalize"?
        """
        self.SetupNetworks(envWrapper, hiddenSize)
        self.SetupOptimizers(actorLearningRate, criticLearningRate)
        self.SetupNoiseProcess(envWrapper, 0, noiseShift, noiseScale)
        self.experiences = Memory(maxMemorySize)

    def Update(self):
        return self.__UpdateWithReplay__()

    def SaveExperience(self, state, action, nextState, reward, done):
        # TODO: [expmt] try spacing these out?
        self.experiences.push(state=state, action=action, reward=reward, next_state=nextState, done=done)

    def GetBestAction(self, state):
        state = Variable(torch.from_numpy(state).float())
        action = self.actor.forward(state)
        action = action.detach().numpy()
        return action

    def GetAction(self, state, offset=0):
        assert(self.noiseProcess is not None)
        action = self.GetBestAction(state)
        action = self.noiseProcess.get_action(action, offset)
        return action

    def GetValue(self, state, action):
        s = torch.FloatTensor(state).unsqueeze(0)
        a = torch.FloatTensor(action).unsqueeze(0)
        return self.critic.forward(s, a).detach().squeeze(0).numpy()[0]

    def SaveLearntModel(self, filepath):
        DdpgAgent.__SaveNnModelsAndOptimizers__(filepath, self)

    def LoadLearntModel(self, filepath):
        raise NotImplementedError

#
# Internal functions
#

    def SetupNetworks(self, envWrapper, hiddenSize):
        assert(0 <= hiddenSize)

        numStates = envWrapper.ObservationSpaceN()
        numActions = envWrapper.ActionSpaceN()
        self.actor = Actor(input_size=numStates, output_size=numActions, hidden_size=hiddenSize)
        self.actorTarget = Actor(input_size=numStates, output_size=numActions, hidden_size=hiddenSize)
        self.critic = Critic(input_size=numStates + numActions, output_size=1, hidden_size=hiddenSize)
        self.criticTarget = Critic(input_size=numStates + numActions, output_size=1, hidden_size=hiddenSize)

        # We initialize the target networks as copies of the original networks
        for targetParam, param in zip(self.actorTarget.parameters(), self.actor.parameters()):
            targetParam.data.copy_(param.data)
        for targetParam, param in zip(self.criticTarget.parameters(), self.critic.parameters()):
            targetParam.data.copy_(param.data)

        # TODO: random actor/critic weight initialization (Javier?)

    def SetupOptimizers(self, actorLearningRate, criticLearningRate):
        assert(0 <= actorLearningRate)
        assert(0 <= criticLearningRate)

        if self.actor and self.critic:
            self.actorOptimizer = torch.optim.Adam(self.actor.parameters(), lr=actorLearningRate)
            self.criticOptimizer = torch.optim.Adam(self.critic.parameters(), lr=criticLearningRate)
            return True
        else:
            return False

    def SetupNoiseProcess(self, envWrapper, mu=0, sigma=0.3, theta=0.15):
        self.noiseProcess = OUStrategy(envWrapper.env.action_space, mu, min_sigma=sigma, theta=theta)

    def __UpdateWithReplay__(self):
        assert(self.experiences is not None)
        assert(self.actor is not None and self.actorOptimizer is not None)

        states, actions, rewards, nextStates, status = self.experiences.sample(self.batchSize)

        if len(states) == 0:
            # Insufficient experience for sampling batch
            return False

        states = torch.FloatTensor(states)
        actions = torch.FloatTensor(actions)
        rewards = torch.FloatTensor(rewards)
        nextStates = torch.FloatTensor(nextStates)

        # Compute actor/critic losses from batches
        learntActions = self.actor.forward(states)
        learntValues = self.critic.forward(states, actions)
        targetActions = self.actorTarget.forward(nextStates).detach()
        targetValues = rewards + self.gamma * self.criticTarget.forward(nextStates, targetActions)
        criticLoss = self.lossFunction(learntValues, targetValues)
        actorLoss = self.critic.forward(states, learntActions)
        # Use the average of all batched losses
        actorLoss = -1 * actorLoss.mean()

        # Run actor/critic optimizers
        self.criticOptimizer.zero_grad()
        self.actorOptimizer.zero_grad()
        criticLoss.backward()
        actorLoss.backward()
        self.criticOptimizer.step()
        self.actorOptimizer.step()

        # "soft" update: new target parameters as weighted average of learnt and target parameters
        for targetParam, param in zip(self.actorTarget.parameters(), self.actor.parameters()):
            targetParam.data.copy_((1 - self.tau) * targetParam.data + self.tau * param.data)
        for targetParam, param in zip(self.criticTarget.parameters(), self.critic.parameters()):
            targetParam.data.copy_((1 - self.tau) * targetParam.data + self.tau * param.data)

        return True

    def __UpdateWithoutReplay__(self):
        raise NotImplementedError

#
# Auxiliary functions
#

    @staticmethod
    def __SaveNnModelsAndOptimizers__(filepath, ddpga):
        """
        https: // pytorch.org / tutorials / beginner / saving_loading_models.html  # saving-loading-model-for-inference
        """
        raise NotImplementedError

    @staticmethod
    def __LoadNnModelsAndOptimizers__(filepath):
        raise NotImplementedError

    def Train(self, envWrapper, gamma, tau, hiddenSize, actorLearningRate, criticLearningRate, batchSize, verbose=True):
        """
        args:
            envWrapper     envWrapperwrapper  a wrapper containing gym envWrapperironment
            gamma   float       reward discount rate
            tau     float       soft update rate for target networks
            hSize   int         width of all hidden layers
            alRate  float       learning rate for actor optimization
            clRate  float       learning rate for critic optimization
            bSize   int         sample size for updates using replay

        """

        # TODO: could create generic training function for all agents; a reason to mv args to ctor
        #       as a result, the caller could have better control over training, e.g., interrupt.
        gamma = min(max(0, gamma), 1)
        tau = min(max(0, tau), 1)

        self.SetupNetworks(envWrapper, hiddenSize)
        self.SetupOptimizers(actorLearningRate, criticLearningRate)
        self.SetupNoiseProcess(envWrapper)

        episodicMetrics = []
        globalStart = timer()
        for i in np.arange(self.maxEpisodes):
            epoch = 0
            totalReward = 0
            frames = []
            done = False
            state = envWrapper.Reset()
            start = timer()
            while not done and epoch < self.maxEpochs:
                action = self.GetAction(state, shouldAddNoise=True)                           # TODO: need to "normalize"? hmmm :/
                nextState, reward, done, _ = envWrapper.Step(action)

                self.experiences.push(state, action, reward, nextState, done)   # TODO: [expmt] try spacing these out?
                self.UpdateUsingReplay(gamma, tau, batchSize)

                epoch += 1
                totalReward += reward
                frames.append({
                    'frame': envWrapper.Render(),
                    'state': state,
                    'action': action,
                    'reward': reward})

                if verbose and epoch % (self.maxEpochs / 1000) == 0:
                    RefreshScreen(mode="human")
                    s = torch.FloatTensor(state).unsqueeze(0)
                    a = torch.FloatTensor(action).unsqueeze(0)
                    qv = self.critic.forward(s, a).detach().squeeze(0).numpy()[0]
                    print(f"Training\ne={i}\nr={np.max(reward): 0.2f}\nq={qv: .2f}")

            metrics = Metrics(frames, epoch, timer() - start, totalReward, done)
            episodicMetrics.append(metrics)

        return episodicMetrics, timer() - globalStart

    def Test(self, envWrapper, verbose):
        episodicMetrics = []
        globalStart = timer()
        for i in np.arange(self.maxEpisodes):
            epoch = 0
            totalReward = 0
            frames = []
            done = False
            state = envWrapper.Reset()
            start = timer()
            while not done and epoch < self.maxEpochs:
                action = self.GetAction(state, shouldAddNoise=False)
                nextState, reward, done, _ = envWrapper.Step(action)

                epoch += 1
                totalReward += reward
                frames.append({
                    'frame': envWrapper.Render(),
                    'state': state,
                    'action': action,
                    'reward': reward})

                if verbose and epoch % (self.maxEpochs / 1000) == 0:
                    RefreshScreen(mode="human")
                    s = torch.FloatTensor(state).unsqueeze(0)
                    a = torch.FloatTensor(action).unsqueeze(0)
                    qv = self.critic.forward(s, a).detach().squeeze(0).numpy()[0]
                    print(f"Testing\ne={i}\nr={np.max(reward): 0.2f}\nq={qv: .2f}")

            metrics = Metrics(frames, epoch, timer() - start, totalReward, done)
            episodicMetrics.append(metrics)

        return episodicMetrics, timer() - globalStart
