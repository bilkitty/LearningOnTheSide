import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable

from environments import *
from memory import *

# TODO: after reorg, undo qlearning dependence


class Actor(nn.Module):
    def __init__(self, num_states, num_actions, hidden_size, learning_rate=3e-4):
        nn.Module.__init__(self)

        self.num_actions = num_actions

        self.linear1 = nn.Linear(num_states, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, num_actions)
    
    def forward(self, state):
        """
        args:
            state       torch tensor    contains observed features
        returns:
                        torch tensor    action based on learnt policy
        """
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        # TODO: Why is this a good choice?
        action = torch.tanh(self.linear3(x))
        #policyDistro = F.softmax(self.linear3(x), dim=1)

        return action


class Critic(nn.Module):
    def __init__(self, num_states, num_actions, hidden_size, learning_rate=3e-4):
        nn.Module.__init__(self)
        self.linear1 = nn.Linear(num_states, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, num_actions)

    def forward(self, state, action):
        """
        args:
            state       torch tensor    contains observed features
            action      torch tensor    contains action representation
        returns:
                        float           action value for state-action pair
        """

        stateAction = torch.cat([state, action], 1)
        x = F.relu(self.linear1(stateAction))
        x = F.relu(self.linear2(x))
        value = self.linear3(x)

        return value

# TODO: memory buffer for action replay process
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
        self.lossFunction = lambda x, y: nn.MSELoss(x, y)

    def SetupNetworks(self, env, hiddenSize):
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
        return action

    def UpdateUsingReplay(self, gamma, batchSize):
        states, actions, rewards, nextStates, status = self.experiences.sample(batchSize)

        # Create tensors from experience buffers
        states = torch.FloatTensor(states)
        actions = torch.FloatTensor(actions)
        rewards = torch.FloatTensor(rewards)
        nextStates = torch.FloatTensor(nextStates)

        # TODO: update value network using mse loss btw learnt and target value
        # Compute critic loss from batches
        values = self.critic.forward(states, actions)
        nextActions = self.actorTarget.forward(nextStates).detach()
        targetValues = rewards + gamma * self.criticTarget.forward(nextStates, nextActions)
        criticLoss = self.lossFunction(values, targetValues)

        # TODO: update policy network using sampled policy grad

        # TODO: update target network params as weighted avg of target and learnt params

    def Train(self, env, gamma, hiddenSize, actorLearningRate, criticLearningRate, batchSize):
        # TODO: could create generic training function for all agents; a reason to mv args to ctor
        #       as a result, the caller could have better control over training, e.g., interrupt.
        gamma = min(max(0, gamma), 1)
        self.SetupNetworks(env, hiddenSize)
        self.SetupOptimizers(actorLearningRate, criticLearningRate)

        for i in np.arange(self.maxEpisodes):
            epoch = 0
            reward = 0
            done = False
            state = env.Reset()
            # TODO: initialize noise process

            while not done and epoch < self.maxEpochs:
                action = self.GetAction(state) # TODO: add some noise

                nextState, reward, done, _ = env.Step(action) # TODO: store in replay buffer: could we try spacing these out?

                self.UpdateUsingReplay(gamma, batchSize)

                epoch += 1

    def Test(self):
        raise NotImplementedError
