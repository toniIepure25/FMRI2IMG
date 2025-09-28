import os
import hydra
from omegaconf import DictConfig, OmegaConf
from .utils.logging import get_logger
from .utils.seed import seed_everything
from .pipelines.baseline_train import run_baseline

log = get_logger(__name__)

@hydra.main(version_base=None, config_path="../../configs", config_name="config")
def main(cfg: DictConfig):
    seed_everything(cfg.get("seed", 1337))
    log.info("Config:\n%s", OmegaConf.to_yaml(cfg))
    os.makedirs(cfg.run.output_dir, exist_ok=True)
    run_baseline(cfg)

if __name__ == "__main__":
    main()
