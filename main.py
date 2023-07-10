import argparse

from hydra.utils import instantiate
from omegaconf import OmegaConf

if __name__ == "__main__":
    # get config path from command line
    parser = argparse.ArgumentParser(description="Train the model")
    # Add arguments to the parser
    parser.add_argument("config", type=str, help="Path to the config file")
    parser.add_argument(
        "config_secondary",
        type=str,
        help="Path to the secondary config file such as a test config (will be merged with the main config and override the main config)))",
    )
    # Parse the arguments
    args = parser.parse_args()

    # load config with omegaconf and instantiate trainer with hydra
    # TODO: use for loop to load multiple configs (any number of configs)
    config_path = args.config
    config = OmegaConf.load(config_path)
    if args.config_secondary is not None:
        config_secondary = OmegaConf.load(args.config_secondary)
        config = OmegaConf.merge(config, config_secondary)

    trainer = instantiate(config)
    config_dict = OmegaConf.to_container(config, resolve=True)
    trainer.train(config=config_dict)
