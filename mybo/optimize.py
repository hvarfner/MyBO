import os
import warnings
import contextlib

from typing import Union, List, Dict, Tuple
import hydra
from omegaconf import DictConfig, OmegaConf
from utils.config import parse_parameters, parse_objectives
from registry.strategy import get_generation_strategy

from evaluate.evaluate import mock_evaluate_mo
from ax.service.ax_client import AxClient
from utils.saving import save_run, AX_NAME, suppress_stdout_stderr


# TODO consider transferring data between clients

@hydra.main(version_base=None, config_path="../configs", config_name="base")
def main(cfg : DictConfig) -> None:
    # Prints out the settings that are run in a nice format...
    print(OmegaConf.to_yaml(cfg))
    
    ax_client = get_or_instantiate(cfg)
    # TODO either entire loop or single eval/call? cfg.closed_loop/open_loop?

    for opt_round in range(cfg.max_rounds):
        designs = get_designs(max_num_designs=cfg.batch_size, client_path=cfg.save_path)
        results = [mock_evaluate_mo(design, cfg.task.objectives)
                    for design in designs
        ]
        register_results(results, client_path=cfg.save_path)


# TODO move to interface.py

def get_or_instantiate(cfg: DictConfig) -> AxClient:
    if len(cfg.resume_path) > 0:
        client = get_client(cfg.resume_path)
    elif os.path.isfile(cfg.save_path + AX_NAME):
        if cfg.override:
            client = instantiate_client(cfg)
        else:
            raise SystemExit(f'Not overriding existing client at {cfg.save_path}{AX_NAME}.'
                '\nSet override=1 in the command line to override anyway. Exiting.'
            )
    else:
        client = instantiate_client(cfg)
    return client


def instantiate_client(cfg: DictConfig) -> AxClient:
    # If we enter a path to a run to resume, try and do so (should be a JSON)
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
        parameter_constraints=cfg.task.get('constraints', None),
        overwrite_existing_experiment=True
    )
    save_run(cfg.save_path, ax_client)
    return ax_client


def get_client(client_path: str):
    #with warnings.catch_warnings():
    #    warnings.simplefilter("ignore", category=FutureWarning)
    with suppress_stdout_stderr():
        ax_client = AxClient.load_from_json_file(client_path)
    return ax_client


def get_designs(
    max_num_designs: int = 1, 
    client: Union[AxClient, None] = None, 
    client_path: Union[str, None] = None,
) -> List[Tuple[Dict[str, float], str]]:
    # (trial_index, {x_1: 0.5, x_2: 2.3, ...})
    if client is None:
        client = get_client(client_path + AX_NAME)

    batch_array = []
    for _ in range(max_num_designs):
        # trial contains both the parameters and the index of the trial
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=FutureWarning)
            trial = client.get_next_trial()
        batch_array.append((trial))
        save_run(client_path, client)
    return batch_array



def register_results(
    results: List[Tuple[Dict[str, float], str]],
    client: Union[AxClient, None] = None, 
    client_path: Union[str, None] = None,
) -> None:
    # (trial_index, {coul_eff: 0.5})
    # TODO save run
    if client is None:
        client = get_client(client_path + AX_NAME)
    
    for result in results:
        client.complete_trial(*result)
        save_run(client_path, client)


def fail_pending_trials(
    client: Union[AxClient, None] = None, 
    client_path: Union[str, None] = None,
) -> None:
    pass
if __name__ == '__main__':
    main()