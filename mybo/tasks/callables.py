from typing import Dict, Optional, Tuple

import torch
from botorch.test_functions import SyntheticTestFunction


# TODO make a registry for these

def evaluate_test_function(
    test_function: SyntheticTestFunction, 
    parameters: Dict[str, float], 
    trial_index: str, 
    seed: Optional[int] = 0,
) -> Tuple[Dict[str, float], str]:
    """All test functions simply take x_1, ..., x_n as input and output y1, ..., y_m.

    Args:
        parameters (Dict[str, float]): dict of parameters and their associated value.
        seed (int, optional): If returned with noise, fix the noise randomness.

    Returns:
        _type_: _description_
    """
    x = torch.tensor(
        [[parameters[f"x{i+1}"] for i in range(test_function.dim)]])
    eval = test_function(x)
    # flip the sign if negated
    
    noiseless_eval = (-1) ** test_function.negate * test_function.evaluate_true(x)
    output_dict = {f'y{m + 1}': e.item() for m, e in enumerate(eval.T)}
    output_dict.update({f'f{m + 1}': e.item() for m, e in enumerate(noiseless_eval.T)})
    return output_dict, trial_index
    

def evaluate_benchsuite_funciton(
    function: callable,
    parameters: Dict[str, float], 
    trial_index: str, 
    seed: Optional[int] = 0,
) -> Tuple[Dict[str, float], str]:
    pass