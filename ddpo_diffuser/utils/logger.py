from torch.utils.tensorboard import SummaryWriter

class Logger:
    def __init__(self, config, experiment_label):
        self.bucket = config['defaults']['logger_cfgs']['log_dir']
        self.label = str(experiment_label)
        self.writer = SummaryWriter(self.bucket + '/' + self.label)

    def write(self, label, value, epoch):
        self.writer.add_scalar(label, value, epoch)
