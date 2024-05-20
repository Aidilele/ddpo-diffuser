from ddpo_diffuser.utils.builder import (
    build_diffuser,
    build_env,
    build_noise_model,
    build_evaluator,
    build_dataset,
    build_config)


def evaluate():
    config_path = "C:\Project\ddpo-diffuser\ddpo_diffuser/runs/2024-5-20-12-43-39"
    config = build_config(config_path=config_path)
    env = build_env(config=config)
    dataset = build_dataset(config=config)
    noise_model = build_noise_model(config=config, env=env)
    diffuser = build_diffuser(config=config, noise_model=noise_model, env=env)
    evaluator = build_evaluator(config=config, diffuser_model=diffuser, env=env, dataset=dataset)
    evaluator.eval()


if __name__ == '__main__':
    evaluate()
