import argparse

from hydra.utils import instantiate
from omegaconf import OmegaConf


def read_configs_and_instantiate():
    # get config path from command line
    parser = argparse.ArgumentParser(description="Train the model")
    # Add arguments to the parser
    parser.add_argument("configs", type=str, nargs="+", help="Path to the config file")
    # Parse the arguments
    args = parser.parse_args()

    # load config with omegaconf and instantiate trainer with hydra
    config_paths = args.configs
    config = OmegaConf.load(config_paths[0])
    for config_path in config_paths[1:]:
        config_secondary = OmegaConf.load(config_path)
        config = OmegaConf.merge(config, config_secondary)

    return instantiate(config), config
