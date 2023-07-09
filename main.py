import sys
from hydra.utils import instantiate
from omegaconf import OmegaConf

if __name__ == "__main__":
    # get config path from command line
    # load config with omegaconf and instantiate trainer with hydra
    config_path = sys.argv[1]
    config = OmegaConf.load(config_path)
    trainer = instantiate(config)
    config_dict = OmegaConf.to_container(config, resolve=True)
    trainer.train(config=config_dict)
