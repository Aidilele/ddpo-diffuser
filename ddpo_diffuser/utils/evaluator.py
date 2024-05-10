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

    def eval(self):
        returns = torch.tensor([[0.9]], device=self.device)
        for episode in range(self.evalutate_episode):
            obs = self.env.reset()
            self.env.render()
            terminal = False
            ep_reward = 0
            obs = torch.from_numpy(obs).unsqueeze(0).to(self.device)
            obs = self.dataset.normalizer.normalize(obs)
            frames = []
            step = 0
            while (step < self.episode_max_length) and (not terminal):
                # for i in range(self.episode_max_length):
                x = self.model.conditional_sample(obs, returns=returns)
                pred_action_obs = x[:, self.obs_history_length - 1:]
                obs = pred_action_obs[:, 0]
                for pred_step in range(self.multi_step_pred):

                    obs_next = pred_action_obs[:, pred_step + 1]
                    obs_comb = torch.cat((obs, obs_next), dim=-1)
                    pred_action = self.model.inv_model(obs_comb)
                    action = pred_action.squeeze().detach().cpu().numpy()
                    next_obs, reward, terminal, _ = self.env.step(action)
                    frames.append(self.env.render(mode="rgb_array"))
                    ep_reward += reward
                    step += 1
                    if terminal:
                        break
                    # obs = torch.from_numpy(next_obs)
                    obs = torch.tensor(next_obs, dtype=torch.float32, device=self.device).unsqueeze(0)
                    obs = self.dataset.normalizer.normalize(obs)
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
        size = (frames[0].shape[:-1])
        video_path = os.path.join(self.bucket, 'video')
        if not os.path.exists(video_path):
            os.makedirs(video_path)
        video_name = video_path+'/Episode' + str(episode) + '_R' + str(int(ep_reward)) + '.mp4'
        out = cv2.VideoWriter(video_name, cv2.VideoWriter_fourcc(*'mp4v'), 30, size)
        for i in range(len(frames)):
            out.write(frames[i])
        out.release()
