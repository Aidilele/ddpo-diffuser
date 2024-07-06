from ddpo_diffuser.utils.builder import (
    build_diffuser,
    build_env,
    build_noise_model,
    build_evaluator,
    build_dataset,
    build_config,
    build_logger,
    build_inverse_dynamic_model,
    build_diffusion,
)


def evaluate():
    config_path = "C:\Project\ddpo-diffuser\ddpo_diffuser/runs/2024-7-6-17-29-43"
    config = build_config(config_path=config_path)
    env = build_env(config=config)
    dataset = build_dataset(config=config)
    noise_model = build_noise_model(config=config)
    inv_model = build_inverse_dynamic_model(config=config)
    diffuser = build_diffusion(config=config, denoise_model=noise_model, inv_model=inv_model,
                               timestep_respacing='')
    evaluator = build_evaluator(config=config, diffuser_model=diffuser, env=env, dataset=dataset)
    evaluator.eval()


if __name__ == '__main__':
    evaluate()
