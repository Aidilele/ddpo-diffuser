from ddpo_diffuser.model.diffusion import GaussianInvDynDiffusion
from ddpo_diffuser.model.temporal import TemporalUnet
from ddpo_diffuser.model.onlinediffuser import OnlineDiffuser
from ddpo_diffuser.dataset.diffuser_dataset import DeciDiffuserDataset
from ddpo_diffuser.utils.diffuser_trainer import DiffuserTrainer
from ddpo_diffuser.utils.evaluator import Evaluator
from ddpo_diffuser.utils.ReadFiles import load_yaml
from ddpo_diffuser.dataset.rlbuffer import RLBuffer
from ddpo_diffuser.env.environment import ParallelEnv
from ddpo_diffuser.model.dit_model import DiT1d
from ddpo_diffuser.utils.logger import Logger
import torch
import gym
import time
import json
import os


def build_env(config):
    env_name = config['defaults']['env_name']
    parallel_num = config['defaults']['env_parallel_num']

    env = ParallelEnv(env_name=env_name, parallel_num=parallel_num)
    return env


def build_config(config_path=None):
    if config_path == None:
        config_path = "./config/diffuser_config.yaml"
        config = load_yaml(config_path)
        time_info = ''
        for x in list(time.localtime())[:-3]:
            time_info += (str(x) + '-')
        time_info = time_info[:-1]
        bucket = './runs/' + time_info
        config['defaults']['logger_cfgs']['log_dir'] = bucket
        config_save = json.dumps(config, indent=4)
        if not os.path.exists(bucket):
            os.makedirs(bucket)
        config_file_name = bucket + '/' + 'config.json'
        with open(config_file_name, "w", encoding='utf-8') as f:  ## 设置'utf-8'编码
            f.write(config_save)
    else:
        config_file_name = config_path + '/' + 'config.json'
        with open(config_file_name, "r", encoding='utf-8') as f:
            config = json.load(f)

    return config


def build_logger(config, experiment_label):
    logger = Logger(config=config, experiment_label=experiment_label)
    return logger


def build_noise_model(config, env):
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    if config['defaults']['algo_cfgs']['noise_model'] == 'TemporalUnet':
        noise_model = TemporalUnet(
            horizon=config['defaults']['algo_cfgs']['horizon'],
            transition_dim=obs_dim,
            dim=config['defaults']['model_cfgs']['temporalU_model']['dim'],
            dim_mults=config['defaults']['model_cfgs']['temporalU_model']['dim_mults'],
            returns_condition=config['defaults']['dataset_cfgs']['include_returns'],
            calc_energy=config['defaults']['model_cfgs']['temporalU_model']['calc_energy'],
            condition_dropout=config['defaults']['model_cfgs']['temporalU_model']['condition_dropout'],
        )
    elif config['defaults']['algo_cfgs']['noise_model'] == 'DiT':
        noise_model = DiT1d(
            x_dim=obs_dim,
            action_dim=action_dim,
            cond_dim=config['defaults']['model_cfgs']['DiT']["cond_dim"],
            hidden_dim=config['defaults']['model_cfgs']['DiT']['hidden_dim'],
            n_heads=config['defaults']['model_cfgs']['DiT']['n_heads'],
            depth=config['defaults']['model_cfgs']['DiT']['depth'],
            dropout=config['defaults']['model_cfgs']['DiT']['dropout'],
        )

    return noise_model


def build_diffuser(config, noise_model, env):
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    diffuser = GaussianInvDynDiffusion(
        model=noise_model,
        horizon=config['defaults']['algo_cfgs']['horizon'],
        observation_dim=obs_dim,
        action_dim=action_dim,
        n_timesteps=config['defaults']['algo_cfgs']['n_diffusion_steps'],
        clip_denoised=config['defaults']['model_cfgs']['diffuser_model']['clip_denoised'],
        predict_epsilon=config['defaults']['model_cfgs']['diffuser_model']['predict_epsilon'],
        hidden_dim=config['defaults']['model_cfgs']['diffuser_model']['hidden_dim'],
        loss_discount=config['defaults']['model_cfgs']['diffuser_model']['loss_discount'],
        returns_condition=config['defaults']['dataset_cfgs']['include_returns'],
        condition_guidance_w=config['defaults']['model_cfgs']['diffuser_model']['condition_guidance_w'],
        train_only_inv=config['defaults']['model_cfgs']['diffuser_model']['train_only_inv'],
        history_length=config['defaults']['train_cfgs']['obs_history_length'],
        multi_step_pred=config['defaults']['evaluate_cfgs']['multi_step_pred'],
    )
    return diffuser


def build_dataset(config):
    dataset = DeciDiffuserDataset(
        dataset_name=config['defaults']['train_cfgs']['dataset'],
        batch_size=config['defaults']['algo_cfgs']['batch_size'],
        device=torch.device(config['defaults']['train_cfgs']['device']),
        horizon=config['defaults']['algo_cfgs']['horizon'],
        include_returns=config['defaults']['dataset_cfgs']['include_returns'],
    )
    return dataset


def build_rlbuffer(config, env):
    x_dim = env.observation_space.shape[0]
    rlbuffer = RLBuffer(
        config=config,
        x_dim=x_dim
    )
    return rlbuffer


def build_trainer(config, diffuser_model, dataset, logger):
    trainer = DiffuserTrainer(diffuser_model=diffuser_model,
                              dataset=dataset,
                              logger=logger,
                              total_steps=config['defaults']['train_cfgs']['total_steps'],
                              train_lr=config['defaults']['train_cfgs']['lr'],
                              gradient_accumulate_every=config['defaults']['train_cfgs']['gradient_accumulate_every'],
                              save_freq=config['defaults']['logger_cfgs']['save_model_freq'],
                              train_device=config['defaults']['train_cfgs']['device'],
                              bucket=config['defaults']['logger_cfgs']['log_dir']
                              )

    return trainer


def build_evaluator(config, diffuser_model, env, dataset):
    evaluator = Evaluator(
        config=config,
        diffuser_model=diffuser_model,
        env=env,
        dataset=dataset
    )
    return evaluator


def build_online_diffuser(config, diffuser_model, env, dataset, rlbuffer, logger):
    online_diffuser = OnlineDiffuser(
        config=config,
        env=env,
        diffuser=diffuser_model,
        dataset=dataset,
        rlbuffer=rlbuffer,
        logger=logger
    )
    return online_diffuser
