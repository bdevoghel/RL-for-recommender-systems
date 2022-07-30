import numpy as np
import torch
import random
from gym.spaces import Box
from collections import namedtuple

from utils.ddpg import DDPG
from utils.td3 import Agent as TD3


"""
From: https://github.com/ikostrikov/pytorch-ddpg-naf
"""


Transition = namedtuple(
    "Transition", ("state", "action", "reward", "next_state", "has_not_left")
)

class Agent:
    def select_action(self, state, iter):
        raise NotImplementedError()

    def remember(self, state, action, reward, next_state, has_not_left):
        raise NotImplementedError()
    
    def learn(self):
        raise NotImplementedError()

    def __repr__(self):
        return f"{self.__class__.__name__}()"


class DDPGAgent(Agent):
    def __init__(
        self,
        observations_space: Box,
        action_space: Box,
        memory_sample_size=128,
        alpha=0.0001,
        beta=0.001,
        gamma=0.95,
        tau=0.001,
        hidden_size=128,
        ounoise_sigma=0.2,
        ounoise_sigma_min=0.0,
        ounoise_decay_period=None,
    ):
        self.observations_space = observations_space
        self.action_space = action_space
        self.updates = 0  # counter of number of updates to parameters

        self.memory = ReplayBuffer(capacity=1e6)
        self.memory_sample_size = memory_sample_size

        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.tau = tau
        self.hidden_size = hidden_size

        self.ddpg = DDPG(
            alpha=alpha,
            beta=beta,
            gamma=gamma,
            tau=tau,
            hidden_size=hidden_size,
            num_inputs=observations_space.shape[0],
            action_space=action_space,
        )

        self.ounoise = OUNoise(
            action_space.shape[0], 
            sigma=ounoise_sigma,
            sigma_min=ounoise_sigma_min,
            sigma_decay_period=ounoise_decay_period,
        )  # action noise
        self.param_noise = None  # observation noise

    def select_action(self, state, time_step):
        state = torch.Tensor(np.array([state]))
        prediction = self.ddpg.select_action(state, self.ounoise, self.param_noise)
        return prediction.numpy()[0]

    def get_q(self, state, action):
        state = torch.Tensor(np.array([state]))
        action = torch.Tensor(np.array([action]))
        q =  self.ddpg.get_q(state, action)
        return q.detach().numpy()[0,0]

    def remember(self, state, action, reward, next_state, has_not_left):
        self.memory.push(state, action, reward, next_state, has_not_left)
    
    def learn(self):
        if (len(self.memory) > self.memory_sample_size):  # if memory is sufficiently populated
            batch = self.memory.sample(self.memory_sample_size, batchwise=True)

            value_loss, policy_loss = self.ddpg.update_parameters(batch)
            self.updates += 1

    def __repr__(self):
        return f"{self.__class__.__name__}(g{self.gamma}-t{self.tau}-a{self.alpha}-b{self.beta}-hs{self.hidden_size}-sm{self.ounoise.sigma_min})"


class TD3Agent(Agent):
    def __init__(self, 
        alpha,
        beta,
        tau,
        env,
        input_dims,
        gamma=0.99,
        update_actor_interval=2,
        warmup=100,
        n_actions=2,
        max_size=1000000,
        layer1_size=400,
        layer2_size=300,
        batch_size=100,
        noise=0.1):
        self.alpha = alpha
        self.beta = beta
        self.tau = tau
        self.gamma = gamma
        self.hs1 = layer1_size
        self.hs2 = layer2_size

        self.td3 = TD3(alpha=alpha, beta=beta, tau=tau, env=env, input_dims=input_dims, gamma=gamma, update_actor_interval=update_actor_interval, warmup=warmup, n_actions=n_actions, max_size=max_size, layer1_size=layer1_size, layer2_size=layer2_size, batch_size=batch_size, noise=noise)
        self.updates = 0

    def select_action(self, state, time_step):
        return self.td3.choose_action(state, time_step)

    def get_q(self, state, action):
        return self.td3.get_q(state, action)

    def remember(self, state, action, reward, next_state, has_not_left):
        self.td3.remember(state, action, reward, next_state, not has_not_left)
    
    def learn(self):
        self.td3.learn()
        self.updates += 1

    def load_models(self):
        self.td3.load_models()

    def __repr__(self):
        return f"{self.__class__.__name__}(g{self.gamma}-t{self.tau}-a{self.alpha}-b{self.beta}-1hs{self.hs1}-2hs{self.hs2})"


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
        self.theta = theta  # weight of past deviation
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
