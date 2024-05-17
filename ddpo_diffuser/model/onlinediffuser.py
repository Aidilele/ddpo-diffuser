import torch
from ddpo_diffuser.model.diffusion import GaussianInvDynDiffusion
import copy
import os
import numpy as np


class EMA():
    '''
        empirical moving average
    '''

    def __init__(self, beta):
        super().__init__()
        self.beta = beta

    def update_model_average(self, ma_model, current_model):
        for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
            old_weight, up_weight = ma_params.data, current_params.data
            ma_params.data = self.update_average(old_weight, up_weight)

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new


class OnlineDiffuser:

    def __init__(self,
                 config,
                 env,
                 dataset,
                 diffuser: GaussianInvDynDiffusion,
                 rlbuffer,
                 reward_model=None):
        self.env = env
        self.obs_dim = self.env.observation_space.shape[0]
        self.action_dim = self.env.action_space.shape[0]
        self.dataset = dataset
        self.rlbuffer = rlbuffer
        self.device = torch.device(config['defaults']['train_cfgs']['device'])
        self.bucket = config['defaults']['logger_cfgs']['log_dir']
        self.diffuser = diffuser.to(self.device)
        self.old_diffuser = copy.deepcopy(diffuser).to(self.device)
        self.old_diffuser.eval()
        self.ema = EMA(0.995)
        self.n_diffusion_steps = config['defaults']['algo_cfgs']['n_diffusion_steps']
        self.r_discounts = config['defaults']['algo_cfgs']['gamma'] ** (
            torch.arange(self.n_diffusion_steps - 1, -1, -1)).to(self.device)
        self.optimizer = torch.optim.Adam(self.diffuser.parameters(), lr=config['defaults']['train_cfgs']['lr'])
        self.reward_model = reward_model

        self.clip = 0.2
        self.horizon = 64
        self.obs_history_length = 1
        self.multi_step_pred = 10
        self.step = 0
        self.save_freq = 100
        self.update_ema_every = 2
        self.step_start_ema = 100
        self.train_frep = 8
        self.save_checkpoints = True

    def reset_parameters(self):
        self.old_diffuser.load_state_dict(self.diffuser.state_dict())

    def step_ema(self):
        if self.step < self.step_start_ema:
            self.reset_parameters()
            return
        self.ema.update_model_average(self.old_diffuser, self.diffuser)

    def obs_history_queue(self, obs, obs_queue: list):
        obs_queue.append(obs)
        if len(obs_queue) > self.obs_history_length:
            obs_queue.__delitem__(0)
        return np.stack(obs_queue, axis=-2).astype(np.float32)

    def action_history_queue(self, action, action_queue: list):
        action_queue.insert(-1, action)
        if len(action_queue) > self.obs_history_length:
            action_queue.__delitem__(0)
        return np.stack(action_queue, axis=-2).astype(np.float32)

    def rollout(self, obs, action, condition):
        self.step += 1
        sample, x_latent, mean, variance, t = self.old_diffuser.multi_steps_diffusion(obs, action, condition)
        state = x_latent[:, :-1, :, :]
        next_state = x_latent[:, 1:, :, :]
        batch_size, diffusion_steps, *_ = state.shape
        log_prob = torch.distributions.normal.Normal(mean, variance).log_prob(next_state).sum([-2, -1])
        log_prob[:, -1] = 1
        # reward = torch.zeros((batch_size, diffusion_steps))
        # reward[:, 0] = self.reward_model(sample, condition)
        reward = 0
        condition = condition.repeat(diffusion_steps, 1).view(batch_size, diffusion_steps, -1)
        history = torch.cat([action, obs], dim=-1)
        history = history.repeat(diffusion_steps, 1, 1).view(batch_size, diffusion_steps, self.obs_history_length, -1)
        return sample[:, :, :self.action_dim], state, next_state, log_prob, reward, history, condition, t

    def cal_pred_traj_advantage(self, reward, target_reward):
        batch_size, n_diffusion_steps, _ = target_reward.shape
        ep_reward = reward
        advantages = ep_reward * self.r_discounts
        return advantages

    def finetune(self):
        state, next_state, log_prob, advantage, history, condition, t = self.rlbuffer.sample()
        batch_size, n_diffusion_steps, horizon, obs_dim = state.shape
        x_pred, mean, variance = self.diffuser.single_step_diffusion(state.view(-1, horizon, obs_dim),
                                                                     history.view(-1, self.obs_history_length, obs_dim),
                                                                     t.view(-1),
                                                                     condition.view(batch_size * n_diffusion_steps, -1))
        mean = mean.view(batch_size, n_diffusion_steps, horizon, obs_dim)
        variance = variance.view(batch_size, n_diffusion_steps, 1, 1)
        new_dist = torch.distributions.normal.Normal(mean, variance)
        new_log_prob = new_dist.log_prob(next_state).sum([-2, -1])
        new_log_prob[torch.where(t == 0)] = 1.0

        ratio = torch.exp(new_log_prob - log_prob)
        advantage = (advantage - advantage.mean()) / advantage.std()
        loss1 = advantage * ratio
        loss2 = torch.clamp(ratio, 1 - self.clip, 1 + self.clip) * advantage
        loss = -torch.min(loss1, loss2).mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def online_training(self):

        returns = 0.8 * torch.ones((self.env.parallel_num, 1), device=self.device)
        for episode in range(100000):
            obs_history = []
            action_history = [self.env.sample_random_action()]
            obs, terminal = self.env.reset()
            obs = self.obs_history_queue(obs, obs_history)
            action = self.action_history_queue(self.env.sample_random_action(), action_history)
            obs_dim = obs.shape[-1]
            # self.env.render()
            obs = torch.from_numpy(obs).to(self.device)
            action = torch.from_numpy(action).to(self.device)
            batch_size, _, _ = obs.shape
            obs = self.dataset.normalizer.normalize(obs)
            step = 0
            while (step < 1000) and (not terminal.all()):
                x, s, n_s, log_prob, _, history, cond, t = self.rollout(obs=obs, action=action, condition=returns)
                pred_action_queue = x[:, self.obs_history_length - 1:]
                ep_reward = []
                for pred_step in range(self.multi_step_pred):
                    pred_action = pred_action_queue[:, pred_step, :]
                    action = pred_action.detach().cpu().numpy()
                    next_obs, reward, terminal, _ = self.env.step(action)
                    action = self.action_history_queue(action, action_history)
                    next_obs = self.obs_history_queue(next_obs, obs_history)
                    ep_reward.append(reward)
                    step += 1
                    if torch.tensor(terminal).all():
                        break
                    # obs = torch.from_numpy(next_obs)
                    obs = torch.tensor(next_obs, dtype=torch.float32, device=self.device)
                    obs = self.dataset.normalizer.normalize(obs)
                    action = torch.tensor(action, dtype=torch.float32, device=self.device)
                reward = torch.tensor(ep_reward, device=self.device).T
                reward = self.dataset.cal_return(reward)
                advantages = self.cal_pred_traj_advantage(reward, cond)
                self.rlbuffer.store(state=s, n_state=n_s, log_prob=log_prob, advantage=advantages, history=history,
                                    cond=cond, t=t)

                if self.rlbuffer.total > self.rlbuffer.max_size and self.step % self.train_frep == 0:
                    self.finetune()
                if self.step % self.update_ema_every == 0:
                    self.step_ema()
                if self.step % self.save_freq == 0:
                    self.save()

    def save(self):
        '''
            saves model and ema to disk;
            syncs to storage bucket if a bucket is specified
        '''
        data = {
            'step': self.step,
            'model': self.diffuser.state_dict(),
            # 'ema': self.ema_model.state_dict()
        }
        savepath = os.path.join(self.bucket, 'checkpoint')
        os.makedirs(savepath, exist_ok=True)
        # logger.save_torch(data, savepath)
        if self.save_checkpoints:
            savepath = os.path.join(savepath, f'state_{self.step}.pt')
        else:
            savepath = os.path.join(savepath, 'state.pt')
        torch.save(data, savepath)
        print(f'[ utils/training ] Saved model to {savepath}')
