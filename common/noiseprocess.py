import random
import numpy as np
import numpy.random as nr

"""
These processes were adopted from https://github.com/vitchyr/rlkit/tree/master/rlkit/exploration_strategies
"""

# TODO: find Boltzmann exploration implementation


class GaussianStrategy:
    """
    This strategy adds Gaussian noise to the action taken by the deterministic policy.
    Based on the rllab implementation.
    """

    DEFAULT_SIGMA = 0.1

    def __init__(self, action_space, max_sigma=DEFAULT_SIGMA, min_sigma=DEFAULT_SIGMA,
                 decay_period=1000000):
        assert len(action_space.shape) == 1
        self._max_sigma = max_sigma
        self._min_sigma = min_sigma
        self._decay_period = decay_period
        self._action_space = action_space

    def get_action_from_raw_action(self, action, t=None, **kwargs):
        sigma = (
            self._max_sigma - (self._max_sigma - self._min_sigma) *
            min(1.0, t * 1.0 / self._decay_period)
        )
        return np.clip(
            action + np.random.normal(size=len(action)) * sigma,
            self._action_space.low,
            self._action_space.high,
        )


class GaussianAndEpislonStrategy:
    """
    With probability epsilon, take a completely random action.
    with probability 1-epsilon, add Gaussian noise to the action taken by a
    deterministic policy.
    """

    DEFAULT_SIGMA = 0.1

    def __init__(self, action_space, epsilon, max_sigma=DEFAULT_SIGMA, min_sigma=DEFAULT_SIGMA,
                 decay_period=1000000):
        assert len(action_space.shape) == 1
        self._max_sigma = max_sigma
        self._epsilon = epsilon
        self._min_sigma = min_sigma
        self._decay_period = decay_period
        self._action_space = action_space

    def get_action_from_raw_action(self, action, t=None, **kwargs):
        if random.random() < self._epsilon:
            return self._action_space.sample()
        else:
            sigma = self._max_sigma - (self._max_sigma - self._min_sigma) * min(1.0, t * 1.0 / self._decay_period)
            return np.clip(
                action + np.random.normal(size=len(action)) * sigma,
                self._action_space.low,
                self._action_space.high,
                )


class OUStrategy:
    """
    This strategy implements the Ornstein-Uhlenbeck process, which adds
    time-correlated noise to the actions taken by the deterministic policy.
    The OU process satisfies the following stochastic differential equation:
    dxt = theta*(mu - xt)*dt + sigma*dWt
    where Wt denotes the Wiener process
    Based on the rllab implementation.
    """

    DEFAULT_SIGMA = 0.2
    DEFAULT_THETA = 0.15
    DEFAULT_MEAN = 0

    def __init__(
            self,
            action_space,
            mu=DEFAULT_MEAN,
            theta=DEFAULT_THETA,
            max_sigma=DEFAULT_SIGMA,
            min_sigma=DEFAULT_SIGMA,
            decay_period=100000,
    ):
        self.mu = mu                                    # mean noise
        self.theta = theta                              # scale deviation from mean state
        self.sigma = max_sigma                          # scale amount of random shift from avg state
        self._max_sigma = max_sigma
        self._min_sigma = min(min_sigma, max_sigma)
        self._decay_period = decay_period
        self.dim = np.prod(action_space.low.shape)
        self.low = action_space.low
        self.high = action_space.high
        self.state = np.ones(self.dim) * self.mu

    def reset(self):
        self.state = np.ones(self.dim) * self.mu

    def evolve_state(self):
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * nr.randn(len(x))
        self.state = x + dx
        return self.state

    def get_action(self, action, t=0):
        ou_state = self.evolve_state()
        self.sigma = self._max_sigma - (self._max_sigma - self._min_sigma) * min(1.0, t * 1.0 / self._decay_period)
        # generate action with ou noise that is bounded action space (high and low attrib)
        return np.clip(action + ou_state, self.low, self.high)