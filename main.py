import argparse

from hydra.utils import instantiate
from omegaconf import OmegaConf

if __name__ == "__main__":
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

    trainer = instantiate(config)
    
    # It converts the merged configuration config into a regular Python dictionary (config_dict). 
    # The resolve=True parameter ensures that any interpolation or variable references in the configuration are resolved and substituted with their actual values.
    config_dict = OmegaConf.to_container(config, resolve=True)
    trainer.train(config=config_dict)
