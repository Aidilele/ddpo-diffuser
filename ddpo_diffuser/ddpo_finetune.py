from ddpo_diffuser.utils.builder import (
    build_diffuser,
    build_env,
    build_noise_model,
    build_online_diffuser,
    build_dataset,
    build_config,
    build_rlbuffer,
    build_logger
)


def finetune():
    config_path = "C:\Project\ddpo-diffuser\ddpo_diffuser/runs/2024-5-21-16-37-58"
    config = build_config(config_path=config_path)
    env = build_env(config=config)
    logger = build_logger(config=config, experiment_label='finetune')
    dataset = build_dataset(config=config)
    noise_model = build_noise_model(config=config, env=env)
    diffuser = build_diffuser(config=config, noise_model=noise_model, env=env)
    rlbuffer = build_rlbuffer(config=config, env=env)
    online_diffuser = build_online_diffuser(config=config,
                                            diffuser_model=diffuser,
                                            env=env,
                                            dataset=dataset,
                                            rlbuffer=rlbuffer,
                                            logger=logger)
    online_diffuser.online_training()


if __name__ == '__main__':
    finetune()
