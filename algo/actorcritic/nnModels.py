import torch
import torch.nn as nn
import torch.nn.functional as F


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
        #       Yoon's example env, Pendulum, has bounded continuous actions. We need to rethink
        #       the choice of output layer that will work for general envs. Is this possible in
        #       this implementation?
        action = torch.tanh(self.linear3(x))
        # policyDistro = F.softmax(self.linear3(x), dim=1)

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
