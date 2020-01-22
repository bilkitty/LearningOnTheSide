import torch
import torch.nn as nn
import numpy as np
from torch.autograd import Variable
from timeit import default_timer as timer

from environments import *
from memory import *
from metrics import *
from .nnModels import *
from utils import RefreshScreen
from noiseprocess import OUStrategy

# TODO: this version of ddpg only supports continuous envs. The interpretation of action space needs to be compatible
#       with Discrete and Box types. -_-


class DdpgAgent:
    DEFAULT_MAX_EPISODES = 100000
    DEFAULT_MAX_EPOCHS = 100000

    def __init__(self, maxMemorySize, maxEpisodes=DEFAULT_MAX_EPISODES, maxEpochs=DEFAULT_MAX_EPOCHS):
        self.actor = None
        self.actorTarget = None
        self.critic = None
        self.criticTarget = None
        self.actorOptimizer = None
        self.criticOptimizer = None
        self.noiseProcess = None
        self.experiences = Memory(maxMemorySize)
        self.lossFunction = nn.MSELoss()
        self.maxEpisodes = maxEpisodes
        self.maxEpochs = maxEpochs

    def SetupNetworks(self, env, hiddenSize):
        assert(0 <= hiddenSize)

        numStates = env.ObservationSpaceN()
        numActions = env.ActionSpaceN()
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

    def SetupNoiseProcess(self, env, mu=0):
        self.noiseProcess = OUStrategy(env.action_space, mu)

    def GetAction(self, state, offset, shouldAddNoise=True):
        state = Variable(torch.from_numpy(state).float().unsqueeze(0))
        action = self.actor.forward(state)
        action = action.detach().numpy()[0,0]
        # TODO: is this the best action? What specifically is this? Also, use squeeze/unsqueeze?
        if shouldAddNoise:
            action = self.noiseProcess.get_action(action, offset)
        return action

    def UpdateUsingReplay(self, gamma, tau, batchSize):
        assert(0 <= gamma and gamma <= 1)
        assert(0 <= tau and tau <= 1)

        states, actions, rewards, nextStates, status = self.experiences.sample(batchSize)

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
        targetValues = rewards + gamma * self.criticTarget.forward(nextStates, targetActions)
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
            targetParam.data.copy_((1 - tau) * targetParam.data + tau * param.data)
        for targetParam, param in zip(self.criticTarget.parameters(), self.critic.parameters()):
            targetParam.data.copy_((1 - tau) * targetParam.data + tau * param.data)

        return True

    def Update(self):
        # TODO: call some version of update (like above)
        return NotImplemented

    def Train(self, env, gamma, tau, hiddenSize, actorLearningRate, criticLearningRate, batchSize, verbose=True):
        """
        args:
            env     envwrapper  a wrapper containing gym environment
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

        self.SetupNetworks(env, hiddenSize)
        self.SetupOptimizers(actorLearningRate, criticLearningRate)
        self.SetupNoiseProcess(env.env)

        episodicMetrics = []
        globalStart = timer()
        for i in np.arange(self.maxEpisodes):
            epoch = 0
            totalReward = 0
            frames = []
            done = False
            state = env.Reset()
            start = timer()
            while not done and epoch < self.maxEpochs:
                action = self.GetAction(state, 0)                           # TODO: need to "normalize"? hmmm :/

                nextState, reward, done, _ = env.Step(action)
                self.experiences.push(state, action, reward, nextState, done)   # TODO: [expmt] try spacing these out?
                self.UpdateUsingReplay(gamma, tau, batchSize)

                epoch += 1
                totalReward += reward
                frames.append({
                    'frame': env.Render(),
                    'state': state,
                    'action': action,
                    'reward': reward})

                if verbose and epoch % (self.maxEpochs / 1000) == 0:
                    RefreshScreen(mode="human")
                    s = torch.FloatTensor(state).unsqueeze(0)
                    a = torch.FloatTensor(action).unsqueeze(0)
                    qv = 0 #self.critic.forward(s, a).detach().squeeze(0).numpy()[0]
                    print(f"Training\ne={i}\nr={np.max(reward): 0.2f}\nq={qv: .2f}")

            metrics = Metrics(frames, epoch, timer() - start, totalReward, done)
            episodicMetrics.append(metrics)

        return episodicMetrics, timer() - globalStart

    def Test(self, env, verbose):
        episodicMetrics = []
        globalStart = timer()
        for i in np.arange(self.maxEpisodes):
            epoch = 0
            totalReward = 0
            frames = []
            done = False
            state = env.Reset()
            start = timer()
            while not done and epoch < self.maxEpochs:
                action = self.GetAction(state, 0)                           # TODO: need to "normalize"? hmmm :/
                nextState, reward, done, _ = env.Step(action)

                epoch += 1
                totalReward += reward
                frames.append({
                    'frame': env.Render(),
                    'state': state,
                    'action': action,
                    'reward': reward})

                if verbose and epoch % (self.maxEpochs / 1000) == 0:
                    RefreshScreen(mode="human")
                    s = torch.FloatTensor(state).unsqueeze(0)
                    a = torch.FloatTensor(action).unsqueeze(0)
                    qv = 0
                    print(f"Testing\ne={i}\nr={np.max(reward): 0.2f}\nq={qv: .2f}")

            metrics = Metrics(frames, epoch, timer() - start, totalReward, done)
            episodicMetrics.append(metrics)

        return episodicMetrics, timer() - globalStart
