import torch
import torch.nn as nn
import torch.nn.functional as F
from environments import *
# TODO: after reorg, undo qlearning dependence


class Actor(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_size, learning_rate=3e-4):
        nn.Module.__init__(self)

        self.num_actions = num_actions

        self.linear1 = nn.Linear(num_inputs, hidden_size)
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
    def __init__(self, num_inputs, num_actions, hidden_size, learning_rate=3e-4):
        nn.Module.__init__(self)
        self.linear1 = nn.Linear(num_inputs, hidden_size)
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


class DdpgAgent:
    DEFAULT_GAMMA = 0.1
    DEFAULT_MAX_EPISODES = 10
    DEFAULT_MAX_EPOCHS = 1000

    def __init__(self, gamma):
        self.gamma = gamma
        self.actor = None
        self.actorTarget = None
        self.critic = None
        self.criticTarget = None

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

    def SetParameters(self, gamma):
        self.gamma = gamma

    def Train(self, env, hiddenSize=3):

        self.SetupNetworks(env, hiddenSize)
        for i in np.arange(DdpgAgent.DEFAULT_MAX_EPISODES):
            epoch = 0
            reward = 0
            done = False
            state = env.reset()
            while not done:
                actions = self.actor.forward(state) # TODO: add some noise
                value = self.critic.forward(state, actions)

                nextState, reward, done, _ = env.step(actions) # TODO: store in replay buffer
                nextActions = self.actorTarget.forward(nextState)

                # TODO: using samples from replay buffer...
                # update policy networks

                # update value networks
                targetValue = reward + self.gamma * self.criticTarget.forward(nextState, nextActions)
                loss = nn.MSELoss(value, targetValue)
                # TODO: optimizer

    def Test(self):
        raise NotImplementedError
