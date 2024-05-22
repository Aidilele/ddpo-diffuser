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
                 logger,
                 reward_model=None):
        self.config = config
        self.env = env
        self.obs_dim = self.env.observation_space.shape[0]
        self.action_dim = self.env.action_space.shape[0]
        self.dataset = dataset
        self.rlbuffer = rlbuffer
        self.logger = logger
        self.device = torch.device(config['defaults']['train_cfgs']['device'])
        self.bucket = config['defaults']['logger_cfgs']['log_dir']
        self.diffuser = diffuser.to(self.device)
        self.load()  # load pretrain model
        self.old_diffuser = copy.deepcopy(diffuser).to(self.device)
        self.old_diffuser.eval()
        self.ema = EMA(0.995)
        self.n_diffusion_steps = config['defaults']['algo_cfgs']['n_diffusion_steps']
        self.r_discounts = config['defaults']['algo_cfgs']['gamma'] ** (
            torch.arange(self.n_diffusion_steps - 1, -1, -1)).to(self.device)
        self.optimizer = torch.optim.Adam(self.diffuser.model.final_layer.parameters(),
                                          lr=config['defaults']['train_cfgs']['lr'])
        self.reward_model = reward_model

        self.save_freq = config['defaults']['logger_cfgs']['save_model_freq']
        self.horizon = config['defaults']['algo_cfgs']['horizon']
        self.obs_history_length = config['defaults']['train_cfgs']['obs_history_length']
        self.multi_step_pred = config['defaults']['evaluate_cfgs']['multi_step_pred']

        self.episode_max_length = 1000
        self.clip = 0.2
        self.step = 0
        self.update_ema_every = 2
        self.step_start_ema = 100
        self.train_frep = self.rlbuffer.max_size
        self.save_checkpoints = bool(self.save_freq)
        self.train_step = 0

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

    def rollout(self, history, condition):
        self.step += 1
        sample, x_latent, mean, variance, t = self.old_diffuser.multi_steps_diffusion(history, condition)
        state = x_latent[:, :-1, :, :]
        next_state = x_latent[:, 1:, :, :]
        batch_size, diffusion_steps, *_ = state.shape
        log_prob = torch.distributions.normal.Normal(mean, variance).log_prob(next_state).sum([-2, -1])
        log_prob[:, -1] = 1
        reward = 0
        condition = condition.repeat(1, diffusion_steps).view(batch_size, diffusion_steps, -1)
        history = history.repeat(diffusion_steps, 1, 1).view(batch_size, diffusion_steps, self.obs_history_length, -1)
        return sample, state, next_state, log_prob, reward, history, condition, t

    def cal_pred_traj_advantage(self, ep_reward):
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
        self.logger.write('advantage', advantage.mean(), self.train_step)
        # advantage = (advantage - advantage.mean()) / advantage.std()
        loss1 = advantage * ratio
        loss2 = torch.clamp(ratio, 1 - self.clip, 1 + self.clip) * advantage
        loss = -torch.min(loss1, loss2).mean()
        self.logger.write('ratio_loss', loss1.mean(), self.train_step)
        self.logger.write('clip_loss', loss2.mean(), self.train_step)
        self.logger.write('min_loss', loss, self.train_step)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.train_step += 1

    def cal_reward(self, curr_cond, target_cond):
        curr_cond = curr_cond.view(-1, 1)
        punish = -(curr_cond - target_cond) ** 2
        punish[torch.where(torch.abs(curr_cond - target_cond) < 0.1)] += 1
        return punish

    def online_training(self):

        for episode in range(100000):
            obs_history = []
            obs, terminal = self.env.reset()
            obs = self.obs_history_queue(obs, obs_history)
            obs = torch.from_numpy(obs).to(self.device)
            batch_size, _, _ = obs.shape
            obs = self.dataset.normalizer.normalize(obs)
            step = 0
            returns = torch.rand((self.env.parallel_num, 1), device=self.device)
            ep_reward = []
            ep_s = []
            ep_ns = []
            ep_log_prob = []
            ep_history = []
            ep_cond = []
            ep_t = []

            while (step < self.episode_max_length) and (not terminal.all()):
                x, s, n_s, log_prob, _, history, cond, t = self.rollout(history=obs, condition=returns)
                ep_s.append(s)
                ep_ns.append(n_s)
                ep_log_prob.append(log_prob)
                ep_history.append(history)
                ep_cond.append(cond)
                ep_t.append(t)
                pred_queue = x[:, self.obs_history_length - 1:]
                for pred_step in range(self.multi_step_pred):
                    next_obs = pred_queue[:, pred_step + 1]
                    obs_comb = torch.cat([obs[:, -1, :], next_obs], dim=-1)
                    pred_action = self.old_diffuser.inv_model(obs_comb)
                    action = pred_action.detach().cpu().numpy()
                    next_obs, reward, terminal, _ = self.env.step(action)
                    next_obs = self.obs_history_queue(next_obs, obs_history)
                    ep_reward.append(reward)
                    step += 1
                    if terminal.all():
                        break
                    obs = torch.tensor(next_obs, dtype=torch.float32, device=self.device)
                    obs = self.dataset.normalizer.normalize(obs)

                # reward = torch.tensor(ep_reward, device=self.device).T
                # reward = self.dataset.cal_return(reward)
                # advantages = self.cal_pred_traj_advantage(reward, cond)
                # self.rlbuffer.store(state=s, n_state=n_s, log_prob=log_prob, advantage=advantages, history=history,
                #                     cond=cond, t=t)

                if self.rlbuffer.total > self.rlbuffer.max_size and self.step % self.train_frep == 0:
                    for _ in range(self.rlbuffer.max_size // self.rlbuffer.sample_batch_size):
                        self.finetune()
                if self.step % self.update_ema_every == 0:
                    self.step_ema()
                if self.step % self.save_freq == 0:
                    self.save()

            ep_reward = torch.tensor(np.array(ep_reward), device=self.device).T
            ep_length = torch.zeros(len(ep_reward), dtype=torch.int64, device=self.device)
            for index, each_epr in enumerate(ep_reward):
                ep_length[index] = len(each_epr[torch.where(each_epr != 0)]) // self.multi_step_pred
            ep_reward = ep_reward.sum(-1) / self.dataset.best_traj_returns
            ep_reward = self.cal_reward(ep_reward, returns)
            advantage = self.cal_pred_traj_advantage(ep_reward).repeat(len(ep_s), 1).view(len(ep_s), -1,
                                                                                          self.n_diffusion_steps)
            ep_s = torch.stack(ep_s)
            ep_ns = torch.stack(ep_ns)
            ep_log_prob = torch.stack(ep_log_prob)
            ep_history = torch.stack(ep_history)
            ep_cond = torch.stack(ep_cond)
            ep_t = torch.stack(ep_t)
            for index, length in enumerate(ep_length):
                self.rlbuffer.store(
                    state=ep_s[:length, index],
                    n_state=ep_ns[:length, index],
                    log_prob=ep_log_prob[:length, index],
                    advantage=advantage[:length, index],
                    history=ep_history[:length, index],
                    cond=ep_cond[:length, index],
                    t=ep_t[:length, index],
                )

        # self.dataset.cal_returns(ep_reward)

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

    def load(self):
        step = self.config['defaults']['evaluate_cfgs']['evaluate_model_index']
        if step != None:
            loadpath = os.path.join(self.bucket, f'checkpoint/state_{step}.pt')
        else:
            loadpath = os.path.join(self.bucket, f'checkpoint/state.pt')
        data = torch.load(loadpath)
        self.step = data['step']
        self.diffuser.load_state_dict(data['model'])
