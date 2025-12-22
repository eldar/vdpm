from pathlib import Path

from omegaconf import OmegaConf
from hydra import compose
from hydra.core.hydra_config import HydraConfig
from hydra.utils import to_absolute_path


def load_config(cfg):
    """
    Borrowed from: https://stackoverflow.com/a/67172466
    """
    if cfg.load_exp is not None:
        output_dir = Path(to_absolute_path(cfg.load_exp))
        original_overrides = OmegaConf.load(output_dir / ".hydra/overrides.yaml")
        current_overrides = HydraConfig.get().overrides.task

        hydra_config = OmegaConf.load(output_dir / ".hydra/hydra.yaml")
        # getting the config name from the previous job.
        config_name = hydra_config.hydra.job.config_name
        # concatenating the original overrides with the current overrides
        overrides = original_overrides + current_overrides
        # compose a new config from scratch
        cfg = compose(config_name, overrides=overrides)
    return cfg
