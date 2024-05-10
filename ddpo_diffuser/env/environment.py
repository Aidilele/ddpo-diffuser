import gym
import numpy as np


class ParallelEnv(gym.Env):

    def __init__(self, env_name, parallel_num):

        self.env_list = []
        self.parallel_num = parallel_num
        for i in range(parallel_num):
            self.env_list.append(gym.make(env_name))
        self.observation_space = self.env_list[0].observation_space
        self.action_space = self.env_list[0].action_space

    def reset(self, **kwargs):
        results = []
        for env in self.env_list:
            result = env.reset(**kwargs)
            results.append(result)
        return np.stack(results)

    def step(self, action):
        observations = []
        rewards = []
        dones = []
        infos = []
        for i in range(len(self.env_list)):
            observation, reward, done, info = self.env_list[i].step(action[i])
            observations.append(observation)
            rewards.append(reward)
            dones.append(done)
            infos.append(infos)
        return (
            np.stack(observations),
            np.stack(rewards),
            np.stack(dones),
            infos
        )

    def render(self, mode='human'):
        results = []
        for env in self.env_list:
            results.append(env.render(mode=mode))
        return results
