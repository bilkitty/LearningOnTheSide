import torch
import torch.nn as nn
import torch.nn.functional as F

IS_LOW_DIM = True
FINAL_LAYER_INIT_RANGE_LOWD = 3 * 1e-2


class Actor(nn.Module):
    # TODO: is this architecture specific to ddpg?
    def __init__(self, input_size, output_size, hidden_size):
        """
        args:
            input_size  int             state space dims (for this application)
            output_size int             action space dims (for this application)
        """
        nn.Module.__init__(self)
        assert(len(hidden_size) == 2)
        self.linear1 = nn.Linear(input_size, hidden_size[0])
        self.linear2 = nn.Linear(hidden_size[0], hidden_size[1])
        self.linear3 = nn.Linear(hidden_size[1], output_size)

        # Recommended init (Lillicrap, 2015) for final layers
        if IS_LOW_DIM:
            self.linear3.weight.data.uniform_(-1 * FINAL_LAYER_INIT_RANGE_LOWD, FINAL_LAYER_INIT_RANGE_LOWD)
            self.linear3.bias.data.uniform_(-1 * FINAL_LAYER_INIT_RANGE_LOWD, FINAL_LAYER_INIT_RANGE_LOWD)
        else:
            self.linear3.weight.data.uniform_(-0.1 * FINAL_LAYER_INIT_RANGE_LOWD, 0.1 * FINAL_LAYER_INIT_RANGE_LOWD)
            self.linear3.bias.data.uniform_(-0.1 * FINAL_LAYER_INIT_RANGE_LOWD, 0.1 * FINAL_LAYER_INIT_RANGE_LOWD)

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
        #       the choice of output layer that will work for general envs. Is this possible icree
        #       this implementation?
        action = torch.tanh(self.linear3(x))

        # TODO: using softmax here would yield some value in [0, 1] but what internally maps this
        #       val to the continuous action space limits (i.e., action_space.hi, action_space.lo)
        #policyDistro = F.softmax(self.linear3(x), dim=1)

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
        assert(len(hidden_size) == 2)
        self.linear1 = nn.Linear(input_size, hidden_size[0])
        self.linear2 = nn.Linear(hidden_size[0], hidden_size[1])
        self.linear3 = nn.Linear(hidden_size[1], output_size)

        # Recommended init (Lillicrap, 2015) for final layers
        if IS_LOW_DIM:
            self.linear3.weight.data.uniform_(-1 * FINAL_LAYER_INIT_RANGE_LOWD, FINAL_LAYER_INIT_RANGE_LOWD)
            self.linear3.bias.data.uniform_(-1 * FINAL_LAYER_INIT_RANGE_LOWD, FINAL_LAYER_INIT_RANGE_LOWD)
        else:
            self.linear3.weight.data.uniform_(-0.1 * FINAL_LAYER_INIT_RANGE_LOWD, 0.1 * FINAL_LAYER_INIT_RANGE_LOWD)
            self.linear3.bias.data.uniform_(-0.1 * FINAL_LAYER_INIT_RANGE_LOWD, 0.1 * FINAL_LAYER_INIT_RANGE_LOWD)

    def forward(self, state, action):
        """
        args:
            state       torch tensor    contains observed features
            action      torch tensor    contains action representation
        returns:
                        float           action value for state-action pair
        """

        stateAction = torch.cat([state, action], dim=1)
        x = F.relu(self.linear1(stateAction))
        x = F.relu(self.linear2(x))
        value = self.linear3(x)

        return value

# TODO: specify actor and critic optimizers. they should take network params and a learning rate as ctor args
