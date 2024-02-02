from omegaconf import OmegaConf

from dos.utils.framework import read_configs_and_instantiate

if __name__ == "__main__":

    trainer, config = read_configs_and_instantiate()
    
    # It converts the merged configuration config into a regular Python dictionary (config_dict). 
    # The resolve=True parameter ensures that any interpolation or variable references in the configuration are resolved and substituted with their actual values.
    config_dict = OmegaConf.to_container(config, resolve=True)
    trainer.train(config=config_dict)
