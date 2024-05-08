import torch
from ddpo_diffuser.model.diffusion import GaussianInvDynDiffusion


class RLxFDiffuser:

    def __init__(self, diffuser: GaussianInvDynDiffusion, env, reward_model):
        self.diffuser = diffuser
        self.old_diffuser = diffuser
        self.old_diffuser.eval()
        self.env = env
        self.reward_model = reward_model
        self.clip = 0.2

    def rollout(self, obs, condition):
        sample, x_latent, mean, variance, t = self.old_diffuser.multi_steps_diffusion(obs, condition)
        batch_size, diffusion_steps, *_ = x_latent.shape
        state = x_latent[:, :-1, :, :]
        next_state = x_latent[:, 1:, :, :]
        log_prob = torch.distributions.normal.Normal(mean, variance).log_prob(next_state)
        reward = torch.zeros((batch_size, diffusion_steps))
        reward[:, 0] = self.reward_model(sample, condition)


        return (state, next_state, log_prob, reward, condition, t)

    def finetune(self, state, next_state, log_prob, reward, condition, t):
        x_pred, mean, variance = self.diffuser.single_step_diffusion(state, t, condition)
        new_dist = torch.distributions.normal.Normal(mean, variance)
        new_log_prob = new_dist.log_prob(next_state)
        advantage = (reward - reward.mean()) / reward.std()
        ratio = torch.exp(new_log_prob - log_prob)

        loss1 = advantage * ratio
        loss2 = torch.clamp(ratio, 1 - self.clip, 1 + self.clip) * advantage
        loss = -torch.min(loss1, loss2).mean()

        self.diffuser.optim.zero_grad()
        loss.backward()
        self.diffuser.optim.step()

    def emu_update(self):

        for p_swa,p_model in zip(self.old_diffuser.parameters(),self.diffuser.parameters()):
            device=p_swa.device
            p_model_=p_model.detach().to(device)



