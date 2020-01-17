import torch
import torch.nn as nn
import numpy as np
from torch.autograd import Variable

from environments import *
from memory import *
from .nnModels import *

# TODO: noise process for action exploration


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

        self.experiences = Memory(maxMemorySize)
        self.noiseProcess = None
        self.maxEpisodes = maxEpisodes
        self.maxEpochs = maxEpochs
        self.lossFunction = nn.MSELoss()

    def SetupNetworks(self, env, hiddenSize):
        assert(0 <= hiddenSize)

        numStates = env.ObservationSpaceN()
        numActions = env.ActionSpaceN()
        self.actor = Actor(numStates, hiddenSize, numActions)
        self.actorTarget = Actor(numStates, hiddenSize, numActions)
        self.critic = Critic(numStates + numActions, hiddenSize, numActions)
        self.criticTarget = Critic(numStates + numActions, hiddenSize, numActions)

        # We initialize the target networks as copies of the original networks
        for targetParam, param in zip(self.actorTarget.parameters(), self.actor.parameters()):
            targetParam.data.copy_(param.data)
        for targetParam, param in zip(self.criticTarget.parameters(), self.critic.parameters()):
            targetParam.data.copy_(param.data)

        # TODO: random actor/critic weight initialization (Javier?)
        # TODO: init target network weights (copy of above)

    def SetupOptimizers(self, actorLearningRate, criticLearningRate):
        assert(0 <= actorLearningRate)
        assert(0 <= criticLearningRate)

        if self.actor and self.critic:
            self.actorOptimizer = torch.optim.Adam(self.actor.parameters(), lr=actorLearningRate)
            self.criticOptimizer = torch.optim.Adam(self.critic.parameters(), lr=criticLearningRate)
            return True
        else:
            return False

    def GetAction(self, state):
        state = Variable(torch.from_numpy(state).float().unsqueeze(0))
        action = self.actor.forward(state)
        action = action.detach().numpy()[0,0]
        # TODO: is this the best action? What specifically is this?
        return np.array([action])

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

    def Train(self, env, gamma, tau, hiddenSize, actorLearningRate, criticLearningRate, batchSize):
        # TODO: could create generic training function for all agents; a reason to mv args to ctor
        #       as a result, the caller could have better control over training, e.g., interrupt.
        gamma = min(max(0, gamma), 1)
        tau = min(max(0, tau), 1)

        self.SetupNetworks(env, hiddenSize)
        self.SetupOptimizers(actorLearningRate, criticLearningRate)

        for i in np.arange(self.maxEpisodes):
            epoch = 0
            reward = 0
            done = False
            state = env.Reset()
            # TODO: initialize noise process

            while not done and epoch < self.maxEpochs:
                action = self.GetAction(state)                                  # TODO: need to "normalize"? hmmm :/
                nextState, reward, done, _ = env.Step(action)                   # TODO: add some noise to action
                self.experiences.push(state, action, reward, nextState, done)   # TODO: [expmt] try spacing these out?
                self.UpdateUsingReplay(gamma, tau, batchSize)

                epoch += 1

    def Test(self):
        raise NotImplementedError
