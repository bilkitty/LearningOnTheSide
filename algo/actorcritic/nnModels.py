import torch
import torch.nn as nn
import torch.nn.functional as F


class Actor(nn.Module):
    # TODO: is this architecture specific to ddpg?
    def __init__(self, input_size, output_size, hidden_size):
        """
        args:
            input_size  int             state space dims (for this application)
            output_size int             action space dims (for this application)
        """
        nn.Module.__init__(self)
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, output_size)

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
    # TODO: is this architecture specific to ddpg?
    def __init__(self, input_size, output_size, hidden_size):
        """
        args:
            input_size  int             sum of state space and action space dims (for this application)
            output_size int             action space dims (for this application)
        """
        nn.Module.__init__(self)
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, output_size)

    def forward(self, state, action):
        """
        args:
            state       torch tensor    contains observed features
            action      torch tensor    contains action representation
        returns:
                        float           action value for state-action pair
        """

        # TODO: change dim based on whether single action/state were provided instead of multiple
        stateAction = torch.cat([state, action], dim=1)
        x = F.relu(self.linear1(stateAction))
        x = F.relu(self.linear2(x))
        value = self.linear3(x)

        return value

# TODO: specify actor and critic optimizers. they should take network params and a learning rate as ctor args