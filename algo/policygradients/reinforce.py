import sys
import torch
import gym
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
import matplotlib.pyplot as plt

GAMMA = 0.9         # We care about long term rewards
ALPHA = 0.1
EPSILON = 0.1
MAX_NUM_EPISODES = 1000
MAX_STEPS = 10000
SHOULD_RENDER = False

"""
Summary of required packages: 
python3 -m pip install torch, gym, numpy, matplotlib

References
Yoon blog: https://medium.com/@thechrisyoon/deriving-policy-gradients-and-implementing-reinforce-f887949bd63
PyTorch examples: https://github.com/pytorch/examples/blob/master/reinforcement_learning/reinforce.py
"""

#
# Neural Network Setup
# The architecture of this network is the following:
#    a 1 layer network (remember that we don't include the output layer in the count)
#    Data enters a fully connected input layer passes through relu activation. A second fully
#    connected layer takes its outputs and passes through softmax activation to yield probabilities
#    for multiple actions.
# We use Adam (or Adaptive Moments Estimation) optimization because it is memory efficient and requires
# little tuning.
class PolicyNetwork(nn.Module):
    # All of the components of our network are here
    def __init__(self, num_inputs, num_actions, hidden_size, learning_rate=3e-4):
        super(PolicyNetwork, self).__init__()

        self.num_actions = num_actions
        self.linear1 = nn.Linear(num_inputs, hidden_size)
        self.linear2 = nn.Linear(hidden_size, num_actions)
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)

    # The structure of the network is here
    def forward(self, state):
        x = F.relu(self.linear1(state))
        # We use softmax at the last layer in order to output probabilities (regress)
        actionProbs = F.softmax(self.linear2(x), dim=1)
        return actionProbs

    def get_action(self, state):
        state = torch.from_numpy(state).float().unsqueeze(0)
        probs = self.forward(Variable(state))
        highest_prob_action = np.random.choice(self.num_actions, p=np.squeeze(probs.detach().numpy()))
        log_prob = torch.log(probs.squeeze(0)[highest_prob_action])
        return highest_prob_action, log_prob

#
# Network training
#
def update_policy(policy_network, rewards, log_probs):
    discounted_rewards = []

    # Compute discounted future rewards for all time steps
    for t in range(len(rewards)):
        Gt = 0
        pw = 0
        for r in rewards[t:]:
            Gt = Gt + GAMMA ** pw * r
            pw = pw + 1
        discounted_rewards.append(Gt)

    # Normalize discounted rewards: center and confine to some range, in this case the mean
    # Q: why use std instead of max or some other finite measure? Rewards outside of std will
    #    have value 1+ or 1-
    # A: The main purpose of doing this normalization is to avoid unstable/noisy gradients. The
    # subtraction of the mean is the primary action that mitigates high variance in grads.
    discounted_rewards = torch.tensor(discounted_rewards)
    discounted_rewards = (discounted_rewards - discounted_rewards.mean()) / (
                discounted_rewards.std() + 1e-9)

    policy_gradient = []
    for log_prob, Gt in zip(log_probs, discounted_rewards):
        policy_gradient.append(-log_prob * Gt)

    policy_network.optimizer.zero_grad()
    policy_gradient = torch.stack(policy_gradient).sum()
    policy_gradient.backward()
    policy_network.optimizer.step()


def main():
    env = gym.make('CartPole-v0')
    policy_net = PolicyNetwork(env.observation_space.shape[0], env.action_space.n, 128)

    numsteps = []
    avg_numsteps = []
    all_rewards = []

    for episode in range(MAX_NUM_EPISODES):
        state = env.reset()
        log_probs = []
        rewards = []

        for steps in range(MAX_STEPS):
            if SHOULD_RENDER:
                env.render()

            action, log_prob = policy_net.get_action(state)
            new_state, reward, done, _ = env.step(action)
            log_probs.append(log_prob)
            rewards.append(reward)

            if done:
                update_policy(policy_net, rewards, log_probs)
                numsteps.append(steps)
                avg_numsteps.append(np.mean(numsteps[-10:]))
                all_rewards.append(np.sum(rewards))
                if episode % 1 == 0:
                    totalReward = np.round(np.sum(rewards), decimals=3)
                    meanReward = np.round(np.mean(all_rewards[-10:]), decimals=3)
                    info = f"episode: {episode}, total reward: {totalReward},"
                    info += f"average_reward: {meanReward}, length: {steps}\n"
                    sys.stdout.write(info)
                break

            state = new_state

    plt.plot(numsteps)
    plt.plot(avg_numsteps)
    plt.xlabel("Episode")
    plt.ylabel("Time steps")
    plt.show()

if __name__ == "__main__":
    main()