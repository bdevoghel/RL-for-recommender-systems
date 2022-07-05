import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.autograd import Variable


def soft_update(target_network, source_network, tau):
    for target, source in zip(target_network.parameters(), source_network.parameters()):
        target.data.copy_(target.data * (1.0 - tau) + source.data * tau)


def hard_update(target_network, source_network):
    for target, source in zip(target_network.parameters(), source_network.parameters()):
        target.data.copy_(source.data)


class LayerNorm(nn.Module):
    def __init__(self, num_features, eps=1e-5, affine=True):
        super(LayerNorm, self).__init__()
        self.num_features = num_features
        self.affine = affine
        self.eps = eps

        if self.affine:
            self.gamma = nn.Parameter(torch.Tensor(num_features).uniform_())
            self.beta = nn.Parameter(torch.zeros(num_features))

    def forward(self, x):
        shape = [-1] + [1] * (x.dim() - 1)
        mean = x.view(x.size(0), -1).mean(1).view(*shape)
        std = x.view(x.size(0), -1).std(1).view(*shape)

        y = (x - mean) / (std + self.eps)
        if self.affine:
            shape = [1, -1] + [1] * (x.dim() - 2)
            y = self.gamma.view(*shape) * y + self.beta.view(*shape)
        return y


class Actor(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, hidden_size, num_inputs, action_space):
        super(Actor, self).__init__()
        self.action_space = action_space
        num_outputs = action_space.shape[0]

        self.linear1 = nn.Linear(num_inputs, hidden_size)
        self.ln1 = LayerNorm(hidden_size)

        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.ln2 = LayerNorm(hidden_size)

        self.mu = nn.Linear(hidden_size, num_outputs)
        self.mu.weight.data.mul_(0.1)
        self.mu.bias.data.mul_(0.1)

    def forward(self, inputs):
        x = inputs

        x = self.linear1(x)
        x = self.ln1(x)
        x = F.relu(x)

        x = self.linear2(x)
        x = self.ln2(x)
        x = F.relu(x)

        mu = self.mu(x)
        mu = torch.tanh(mu)
        return mu


class Critic(nn.Module):
    """Critic (Value) Model."""

    def __init__(self, hidden_size, num_inputs, action_space):
        super(Critic, self).__init__()
        self.action_space = action_space
        num_outputs = action_space.shape[0]

        self.linear1 = nn.Linear(num_inputs, hidden_size)
        self.ln1 = LayerNorm(hidden_size)

        self.linear2 = nn.Linear(hidden_size + num_outputs, hidden_size)
        self.ln2 = LayerNorm(hidden_size)

        self.V = nn.Linear(hidden_size, 1)
        self.V.weight.data.mul_(0.1)
        self.V.bias.data.mul_(0.1)

    def forward(self, inputs, actions):
        x = inputs

        x = self.linear1(x)
        x = self.ln1(x)
        x = F.relu(x)

        x = torch.cat((x, actions), 1)
        x = self.linear2(x)
        x = self.ln2(x)
        x = F.relu(x)

        V = self.V(x)
        return V


class DDPG(object):
    def __init__(self, gamma, tau, hidden_size, num_inputs, action_space):

        self.num_inputs = num_inputs
        self.action_space = action_space

        self.actor = Actor(hidden_size, self.num_inputs, self.action_space)
        self.actor_target = Actor(hidden_size, self.num_inputs, self.action_space)
        # self.actor_perturbed = Actor(hidden_size, self.num_inputs, self.action_space)  # TODO behaviour to be properly defined
        self.actor_optim = Adam(self.actor.parameters(), lr=1e-4)

        self.critic = Critic(hidden_size, self.num_inputs, self.action_space)
        self.critic_target = Critic(hidden_size, self.num_inputs, self.action_space)
        self.critic_optim = Adam(self.critic.parameters(), lr=1e-3)

        self.gamma = gamma  # discount factor for future rewards
        self.tau = tau  # for soft update of target networks

        # make target Actor and target Critic share same weight as base Actor and Critic
        hard_update(self.actor_target, self.actor)
        hard_update(self.critic_target, self.critic)

    def select_action(self, state, action_noise=None, param_noise=None):
        self.actor.eval()
        if param_noise is None:
            mu = self.actor(Variable(state))
        else:
            assert False, "Behaviour of actor_perturbed to be properly defined"
            mu = self.actor_perturbed(Variable(state))

        self.actor.train()
        mu = mu.data

        if action_noise is not None:
            mu += torch.Tensor(action_noise.noise())

        return mu.clamp(-1, 1)

    def update_parameters(self, batch):
        # store buffer values in Torch Variables
        state_batch = Variable(torch.cat(batch.state))
        action_batch = Variable(torch.cat(batch.action))
        reward_batch = Variable(torch.cat(batch.reward))
        next_state_batch = Variable(torch.cat(batch.next_state))
        has_not_left_batch = Variable(torch.cat(batch.has_not_left))

        ## TRAIN CRITIC
        # predict Q_next with frozen Actor and Critic
        next_action_batch = self.actor_target(next_state_batch)
        next_state_action_values = self.critic_target(
            next_state_batch, next_action_batch
        )

        reward_batch = reward_batch.unsqueeze(1)
        has_not_left_batch = has_not_left_batch.unsqueeze(1)
        expected_state_action_batch = reward_batch + (
            self.gamma * has_not_left_batch * next_state_action_values
        )

        # predict Q with with learning Critic
        self.critic_optim.zero_grad()
        state_action_batch = self.critic(state_batch, action_batch)

        # compute loss and backpropagate
        value_loss = F.mse_loss(state_action_batch, expected_state_action_batch)
        value_loss.backward()
        self.critic_optim.step()

        ## TRAIN ACTOR
        # maximize Q value for learning Actor with frozen Critic
        self.actor_optim.zero_grad()
        policy_loss = -self.critic(
            state_batch, self.actor(state_batch)
        )  # TODO use critic_target ?

        # compute loss and backpropagate
        policy_loss = policy_loss.mean()
        policy_loss.backward()
        self.actor_optim.step()

        ## ...
        # make target Actor and target Critic converge to same weight as base Actor and Critic
        soft_update(self.actor_target, self.actor, self.tau)
        soft_update(self.critic_target, self.critic, self.tau)

        return value_loss.item(), policy_loss.item()

    def perturb_actor_parameters(self, param_noise):
        """Apply parameter noise to actor model, for exploration."""
        assert False, "Behaviour of actor_perturbed to be properly defined"
        hard_update(self.actor_perturbed, self.actor)
        params = self.actor_perturbed.state_dict()
        for name in params:
            if "ln" in name:
                pass
            param = params[name]
            param += torch.randn(param.shape) * param_noise.current_stddev

    def save_model(self, env_name, suffix="", actor_path=None, critic_path=None):
        if not os.path.exists("models/"):
            os.makedirs("models/")

        if actor_path is None:
            actor_path = f"models/ddpg_actor_{env_name}_{suffix}"
        if critic_path is None:
            critic_path = f"models/ddpg_critic_{env_name}_{suffix}"
        print(f"Saving models to {actor_path} and {critic_path}")
        torch.save(self.actor.state_dict(), actor_path)
        torch.save(self.critic.state_dict(), critic_path)

    def load_model(self, actor_path, critic_path):
        print(f"Loading models from {actor_path} and {critic_path}")
        if actor_path is not None:
            self.actor.load_state_dict(torch.load(actor_path))
        if critic_path is not None:
            self.critic.load_state_dict(torch.load(critic_path))
