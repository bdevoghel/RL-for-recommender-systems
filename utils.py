import numpy as np
import torch
import random
from gym.spaces import Box
from collections import namedtuple

from ddpg import DDPG


"""
From: https://github.com/ikostrikov/pytorch-ddpg-naf
"""


Transition = namedtuple(
    "Transition", ("state", "action", "reward", "next_state", "has_not_left")
)
LEARN_ITERATIONS = 1  # number of times the memory is sampled and the parameters are updated per learn() call


class DDPGAgent:
    def __init__(
        self,
        observations_space: Box,
        action_space: Box,
        memory_sample_size=128,
        gamma=0.95,
        tau=0.001,
        hidden_size=128,
        ounoise_decay_period=None,
    ):
        self.observations_space = observations_space
        self.action_space = action_space
        self.updates = 0  # counter of number of updates to parameters

        self.memory = ReplayBuffer(capacity=1e4)
        self.memory_sample_size = memory_sample_size

        self.gamma = gamma
        self.tau = tau
        self.hidden_size = hidden_size

        self.ddpg = DDPG(
            gamma=gamma,
            tau=tau,
            hidden_size=hidden_size,
            num_inputs=observations_space.shape[0],
            action_space=action_space,
        )

        self.ounoise = OUNoise(
            action_space.shape[0], sigma_decay_period=ounoise_decay_period,
        )  # action noise
        self.param_noise = None  # observation noise

    def select_action(self, state):
        state = torch.Tensor(np.array([state]))
        prediction = self.ddpg.select_action(state, self.ounoise, self.param_noise)
        return prediction.numpy()[0]

    def learn(self):
        if (
            len(self.memory) > self.memory_sample_size
        ):  # if memory is sufficiently populated
            for _ in range(LEARN_ITERATIONS):
                batch = self.memory.sample(self.memory_sample_size, batchwise=True)

                value_loss, policy_loss = self.ddpg.update_parameters(batch)
                self.updates += 1

    def __repr__(self):
        return f"{self.__class__.__name__}(g{self.gamma:.2f}-t{self.tau:.4f}-hs{self.hidden_size:03d})"


class ReplayBuffer(object):
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0  # position where to push next element to

    def push(self, *args, transition: Transition = None):
        """Saves a transition."""
        if transition is None:
            transition = Transition(*[torch.Tensor(np.array([a])) for a in args])

        # increase memory space if needed
        if len(self.memory) < self.capacity:
            self.memory.append(None)

        self.memory[self.position] = transition  # save transition
        self.position = int(
            (self.position + 1) % self.capacity
        )  # increment with roll-over

    def sample(self, n, batchwise=False):
        """Returns n elements sampled at random from memory."""
        samples = random.sample(self.memory, n)

        if batchwise:
            return Transition(
                *zip(*samples)
            )  # transpose tuple (list of tuples to tuple of lists)
        else:
            return samples

    def __len__(self):
        return len(self.memory)


class OUNoise:
    """Ornstein-Uhlenbeck process."""

    def __init__(
        self,
        action_dimension,
        scale=0.1,
        mu=0,
        theta=0.15,
        sigma=0.2,
        sigma_min=0.0,
        sigma_decay=0.0,
        sigma_decay_period=None,
    ):
        """Initialize parameters and noise process."""
        self.action_dimension = action_dimension
        self.scale = scale
        self.mu = mu  # mean to which the process converges
        self.theta = theta  #
        self.sigma = sigma  # variance
        self.sigma_min = sigma_min
        self.sigma_decay = sigma_decay
        self.sigma_decay_period = sigma_decay_period
        self.state = np.ones(self.action_dimension) * self.mu

    def reset(self, t=0):
        """Reset the internal state (= noise) to mean (mu)."""
        self.state = np.ones(self.action_dimension) * self.mu
        if self.sigma_decay_period:
            self.sigma = max(
                self.sigma_min,
                self.sigma
                - (self.sigma - self.sigma_min) * min(1.0, t / self.sigma_decay_period),
            )
        else:
            self.sigma = max(self.sigma_min, self.sigma * self.sigma_decay)

    def noise(self):
        """Update internal state and return it as a noise sample."""
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(len(x))
        self.state = x + dx
        return self.state * self.scale
