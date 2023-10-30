import os

import hydra
from omegaconf import DictConfig, OmegaConf
from utils.config import parse_parameters, parse_objectives
from registry.strategy import get_generation_strategy

from evaluate.evaluate import mock_evaluate_mo
from ax.service.ax_client import AxClient
from utils.saving import save_run

@hydra.main(version_base=None, config_path="../configs", config_name="base")
def main(cfg : DictConfig) -> None:
    # Prints out the settings that are run in a nice format...
    print(OmegaConf.to_yaml(cfg))
    
    # If we enter a path to a run to resume, try and do so (should be a JSON)
    if len(cfg.resume_path) > 0:
        ax_client = AxClient.load_from_json_file(cfg.resume_path)
    
    else:
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

    for i in range(20):
        parameters, trial_index = ax_client.get_next_trial()
        ax_client.complete_trial(trial_index=trial_index, raw_data=mock_evaluate_mo(cfg.task.objectives))
        save_run(cfg.save_path, ax_client)

if __name__ == '__main__':
    main()