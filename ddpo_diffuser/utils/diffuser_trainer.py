import copy
import torch
import os


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


class DiffuserTrainer(object):

    def __init__(self,
                 denoise_model,
                 diffuser_model,
                 dataset,
                 logger,
                 total_steps,
                 ema_decay=0.995,
                 train_lr=2e-5,
                 gradient_accumulate_every=2,
                 step_start_ema=2000,
                 update_ema_every=10,
                 # log_freq=100,
                 sample_freq=1000,
                 save_freq=1000,
                 train_device='cuda',
                 bucket=None,
                 save_checkpoints=True,
                 ):
        super().__init__()
        self.denoise_model = denoise_model
        self.diffuser = diffuser_model
        self.ema = EMA(ema_decay)
        self.ema_model = copy.deepcopy(self.denoise_model)
        self.logger = logger
        self.total_steps = total_steps
        self.update_ema_every = update_ema_every
        self.step_start_ema = step_start_ema
        self.sample_freq = sample_freq
        self.save_freq = save_freq
        self.gradient_accumulate_every = gradient_accumulate_every
        self.dataset = dataset
        params = list(self.diffuser.denoise_model.parameters()) + list(self.diffuser.inv_model.parameters())
        self.optimizer = torch.optim.Adam(params, lr=train_lr)
        self.device = torch.device(train_device)
        self.denoise_model.to(self.device)
        self.ema_model.to(self.device)
        self.bucket = bucket
        self.save_checkpoints = save_checkpoints
        self.step = 0
        path = os.path.join(self.bucket, f'checkpoint')
        if os.path.exists(path):
            model_files = os.listdir(path)
            model_file_name = path + '/' + model_files[-1]
            data = torch.load(model_file_name)
            self.denoise_model.load_state_dict(data['model'])
            self.ema_model.load_state_dict(data['model'])
        self.reset_parameters()

    def reset_parameters(self):
        self.ema_model.load_state_dict(self.denoise_model.state_dict())

    def step_ema(self):
        if self.step < self.step_start_ema:
            self.reset_parameters()
            return
        self.ema.update_model_average(self.ema_model, self.denoise_model)

    def train(self):

        for step in range(self.total_steps):
            self.optimizer.zero_grad()
            for _ in range(self.gradient_accumulate_every):
                batch_sample = self.dataset.sample()
                x = batch_sample[0]
                returns = batch_sample[1]
                model_kwargs = dict(returns=returns)
                loss_dict = self.diffuser.training_losses(model=self.denoise_model, x_start=x,
                                                          model_kwargs=model_kwargs)
                # loss, info = self.diffuser.training_losses(*batch_sample)
                loss = loss_dict['loss'].mean()
                loss = loss / self.gradient_accumulate_every
                loss.backward()
            self.optimizer.step()
            # self.logger.write('loss/diffuser', loss_dict['loss'].mean(), step)
            self.logger.write('loss/vb', loss_dict['vb'].mean(), step)
            self.logger.write('loss/mse', loss_dict['mse'].mean(), step)
            self.logger.write('loss/inv', loss_dict['inv'].mean(), step)

            if self.step % self.update_ema_every == 0:
                self.step_ema()
            if step % self.save_freq == 0:
                self.save()
            self.step += 1

    def save(self):
        '''
            saves model and ema to disk;
            syncs to storage bucket if a bucket is specified
        '''
        data = {
            'step': self.step,
            'model': self.denoise_model.state_dict(),
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

    def load(self, step=None):
        if self.save_checkpoints and step != None:
            loadpath = os.path.join(self.bucket, f'checkpoint/state_{step}.pt')
        else:
            loadpath = os.path.join(self.bucket, f'checkpoint/state.pt')
        data = torch.load(loadpath)
        self.step = data['step']
        self.denoise_model.load_state_dict(data['model'])
        self.ema_model.load_state_dict(data['model'])
