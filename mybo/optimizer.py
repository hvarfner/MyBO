
import hydra
from omegaconf import DictConfig, OmegaConf
from utils.config import parse_parameters, parse_objectives
from registry.strategy import get_generation_strategy

from ax.service.ax_client import AxClient

@hydra.main(version_base=None, config_path="../configs", config_name="base")
def main(cfg : DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))
    
    num_dimensions = len(cfg.task.parameters)
    generation_strategy = get_generation_strategy(
        model_cfg=cfg.model, 
        acq_cfg=cfg.acq, 
        acqopt_cfg=cfg.acqopt, 
        init_cfg=cfg.init,
        num_dimensions=num_dimensions,
    )

    ax_client = AxClient(generation_strategy=generation_strategy)
    ax_client.create_experiment(
        name=cfg.experiment_name,
        parameters=parse_parameters(cfg.task.parameters),
        objectives=parse_objectives(cfg.task.objectives),
        overwrite_existing_experiment=True
    )

    
if __name__ == '__main__':
    main()