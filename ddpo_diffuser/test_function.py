import d4rl
import gym
import torch

env=gym.make('Hopper-v2')
dataset_env=gym.make('hopper-expert-v2')
mu = torch.ones((10,3,5,5))
sigma = torch.ones((10,3,5,5))
action = torch.ones((10,3,5,5))
dist = torch.distributions.normal.Normal(mu, sigma)
log_prob = dist.log_prob(action).exp()



print('ok')
