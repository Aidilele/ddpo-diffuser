from ddpo_diffuser.utils.builder import (
    build_diffusion,
    build_dataset,
    build_env,
    build_noise_model,
    build_trainer,
    build_config,
    build_logger
)


def train(config_path):
    config = build_config(config_path=config_path)
    env = build_env(config=config)
    logger = build_logger(config=config, experiment_label='PreTrain')
    dataset = build_dataset(config=config)
    noise_model = build_noise_model(config=config, env=env)
    diffuser = build_diffusion(denoise_model=noise_model, timestep_respacing='')
    trainer = build_trainer(config=config, diffuser_model=diffuser, dataset=dataset, logger=logger)
    trainer.train()


if __name__ == '__main__':
    # config_path = "C:\Project\ddpo-diffuser\ddpo_diffuser/runs/2024-5-9-10-31-17"
    config_path = None
    train(config_path)
