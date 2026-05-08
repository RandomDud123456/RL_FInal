import torch
import torch.nn as nn
import numpy as np
from torch.distributions.categorical import Categorical


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class PPOAgent(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size):
        super().__init__()
        layers = []
        in_size = input_size
        for h_size in hidden_sizes:
            layers.append(layer_init(nn.Linear(in_size, h_size)))
            layers.append(nn.ReLU())
            in_size = h_size
        layers.append(layer_init(nn.Linear(in_size, 1), std=1.0))
        self.critic = nn.Sequential(*layers)
        layers = []
        in_size = input_size
        for h_size in hidden_sizes:
            layers.append(layer_init(nn.Linear(in_size, h_size)))
            layers.append(nn.ReLU())
            in_size = h_size
        layers.append(layer_init(nn.Linear(in_size, output_size), std=0.01))
        self.actor = nn.Sequential(*layers)
        print(self.actor)
        print(self.critic)

    def get_value(self, x):
        return self.critic(x)

    def get_action_and_value(self, x, action=None):
        logits = self.actor(x)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(x)


class SimpleNeuralNet(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size):
        super(SimpleNeuralNet, self).__init__()
        layers = []
        in_size = input_size
        for h_size in hidden_sizes:
            layers.append(nn.Linear(in_size, h_size))
            layers.append(nn.ReLU())
            in_size = h_size
        layers.append(nn.Linear(in_size, output_size))
        layers.append(nn.Softmax(dim=-1))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


def initialize_policy_net(env, cfg):
    hidden_size = cfg["nn_size"]
    input_size = output_size = len(env.P_list)
    policy_net = SimpleNeuralNet(input_size, hidden_size, output_size)
    return policy_net
