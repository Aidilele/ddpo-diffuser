import d4rl
import gym
import torch

import yaml


def load_yaml(path: str):

    path="./config/diffuser_config.yaml"

    with open(path, encoding='utf-8') as file:
        try:
            kwargs = yaml.load(file, Loader=yaml.FullLoader)  # noqa: S506
        except FileNotFoundError as exc:
            raise FileNotFoundError(f'{path} error: {exc}') from exc

    return kwargs

load_yaml('')
env = gym.make('Hopper-v2')
dataset_env = gym.make('hopper-expert-v2')
mu = torch.ones((10, 3, 5, 5))
sigma = torch.ones((10, 3, 5, 5))
action = torch.ones((10, 3, 5, 5))
dist = torch.distributions.normal.Normal(mu, sigma)
log_prob = dist.log_prob(action).exp()

print('ok')
