import numpy as np
import torch
import cv2
import os


class Evaluator:
    def __init__(self,
                 config,
                 diffuser_model,
                 env,
                 dataset,
                 device='cuda',
                 render=False,
                 evaluate_episode=10,
                 episode_max_length=1000,
                 obs_history_length=1,
                 multi_step_pred=20
                 ):
        self.config = config
        self.model = diffuser_model
        self.env = env
        self.dataset = dataset
        self.device = torch.device(device)
        self.render = render
        self.evalutate_episode = evaluate_episode
        self.episode_max_length = episode_max_length
        self.obs_history_length = obs_history_length
        self.multi_step_pred = multi_step_pred
        self.bucket = config['defaults']['logger_cfgs']['log_dir']
        self.model.eval()
        self.ema_model = self.model
        self.load()
        self.model.to(self.device)
        self.env.reset()

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

    def eval(self):
        returns = 0.8 * torch.ones((self.env.parallel_num, 1), device=self.device)
        for episode in range(self.evalutate_episode):
            obs_history = []
            action_history = [self.env.sample_random_action()]
            obs, terminal = self.env.reset()
            obs = self.obs_history_queue(obs, obs_history)
            action = self.action_history_queue(self.env.sample_random_action(), action_history)
            self.env.render()
            ep_reward = 0
            obs = torch.from_numpy(obs).to(self.device)
            obs = self.dataset.normalizer.normalize(obs)
            action = torch.from_numpy(action).to(self.device)
            frames = []
            step = 0
            while (step < self.episode_max_length) and (not terminal.all()):
                # for i in range(self.episode_max_length):
                x = self.model.conditional_sample(obs, action, returns=returns)
                pred_action_queue = x[:, self.obs_history_length - 1:]
                for pred_step in range(self.multi_step_pred):
                    pred_action = pred_action_queue[:, pred_step, :]
                    action = pred_action.squeeze().detach().cpu().numpy()
                    next_obs, reward, terminal, _ = self.env.step(action)
                    action = self.action_history_queue(action, action_history)
                    next_obs = self.obs_history_queue(next_obs, obs_history)
                    frames.append(self.env.render(mode="rgb_array"))
                    ep_reward += reward
                    step += 1
                    if terminal.all():
                        break
                    # obs = torch.from_numpy(next_obs)
                    obs = torch.tensor(next_obs, dtype=torch.float32, device=self.device)
                    obs = self.dataset.normalizer.normalize(obs)
                    action = torch.tensor(action, dtype=torch.float32, device=self.device)
            self.render_frames(frames, episode, ep_reward)
            print('episode:', episode, '--> ep_reward:', ep_reward)

    def load(self):
        step = self.config['defaults']['evaluate_cfgs']['evaluate_model_index']
        if step != None:
            loadpath = os.path.join(self.bucket, f'checkpoint/state_{step}.pt')
        else:
            loadpath = os.path.join(self.bucket, f'checkpoint/state.pt')
        data = torch.load(loadpath)
        self.step = data['step']
        self.model.load_state_dict(data['model'])

    def render_frames(self, frames, episode, ep_reward):
        frames = np.stack(frames, axis=1)
        size = frames.shape[-3:-1]
        video_path = os.path.join(self.bucket, 'video')
        if not os.path.exists(video_path):
            os.makedirs(video_path)
        for i in range(frames.shape[0]):
            frame=frames[i]
            video_name = video_path + '/Episode' + str(episode) + '_R' + str(int(ep_reward[i])) + '.mp4'
            out = cv2.VideoWriter(video_name, cv2.VideoWriter_fourcc(*'mp4v'), 120, size)
            for j in range(len(frame)):
                out.write(frame[j])
            out.release()
