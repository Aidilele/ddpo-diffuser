import torch


class RLBuffer:
    def __init__(self, config, x_dim, max_size=512, sample_batch_size=64):
        n_diffusion_steps = config['defaults']['algo_cfgs']['n_diffusion_steps']
        horizon = config['defaults']['algo_cfgs']['horizon']
        history_length = config['defaults']['train_cfgs']['obs_history_length']
        self.device = torch.device(config['defaults']['train_cfgs']['device'])
        self.max_size = max_size
        self.sample_batch_size = sample_batch_size
        self.state = torch.zeros((max_size, n_diffusion_steps, horizon, x_dim))
        self.n_state = torch.zeros((max_size, n_diffusion_steps, horizon, x_dim))
        self.log_prob = torch.zeros((max_size, n_diffusion_steps))
        self.advantage = torch.zeros((max_size, n_diffusion_steps))
        self.history = torch.zeros((max_size, n_diffusion_steps, history_length, x_dim))
        self.cond = torch.zeros((max_size, n_diffusion_steps, 1))
        self.t = torch.zeros((max_size, n_diffusion_steps), dtype=torch.int64)
        self.point = 0
        self.total = 0

    def store(self, state, n_state, log_prob, advantage, history, cond, t):
        num_traj = state.shape[0]
        for i in range(num_traj):
            self.state[self.point] = state[i]
            self.n_state[self.point] = n_state[i]
            self.log_prob[self.point] = log_prob[i]
            self.advantage[self.point] = advantage[i]
            self.history[self.point] = history[i]
            self.cond[self.point] = cond[i]
            self.t[self.point] = t[i]
            self.point = (self.point + 1) % self.max_size
            self.total += 1

    def sample(self):
        indices = torch.randint(low=0, high=self.max_size, size=(self.sample_batch_size,), dtype=torch.int64)
        return (
            self.state[indices].to(self.device),
            self.n_state[indices].to(self.device),
            self.log_prob[indices].to(self.device),
            self.advantage[indices].to(self.device),
            self.history[indices].to(self.device),
            self.cond[indices].to(self.device),
            self.t[indices].to(self.device),
        )
