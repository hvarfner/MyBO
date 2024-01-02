from typing import Callable
import torch
from torch import Tensor
from torch.quasirandom import SobolEngine
from botorch.utils.transforms import unnormalize
from botorch.test_functions import SyntheticTestFunction
from gpytorch.mlls import ExactMarginalLogLikelihood
from ax.service.ax_client import AxClient

NUM_SPLITS = 20
def evaluate_mll(ax_client: AxClient, objective: SyntheticTestFunction, num_test_points: int):
    try:
        model = ax_client.get_model_predictions()
    except NotImplementedError:
        return -1e6
    sobol = SobolEngine(len(objective.bounds.T), scramble=True, seed=42)
    test_batch = sobol.draw(num_test_points)
    gp = ax_client.generation_strategy.model.model.surrogate.model
    output = objective.evaluate_true(unnormalize(test_batch, objective.bounds))
    y_transform = ax_client._generation_strategy.model.transforms['StandardizeY']
    objective_name = list(ax_client.experiment.metrics.keys())[0]
    y_mean, y_std = y_transform.Ymean[objective_name], y_transform.Ystd[objective_name]
    mu, _ = ax_client._generation_strategy.model.model.predict(test_batch)
    mu_true = (mu * y_std + y_mean).flatten()

    model = ax_client._generation_strategy.model.model.surrogate.model
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    norm_yvalid = (output - y_mean) / y_std
    norm_yvalid = -norm_yvalid.flatten()

    model.eval()
    res_mll = torch.zeros(num_test_points)
    res_mll = 0
    for i in range(NUM_SPLITS):
        low, high = i * int(num_test_points / NUM_SPLITS), (i + 1)* int(num_test_points / NUM_SPLITS)
        preds = model(test_batch[low:high])
        res_mll += mll(preds, norm_yvalid[low:high])
    res_mll /= NUM_SPLITS
    mean_mll = res_mll.mean().item()
    if len(gp.train_inputs[0]) > 40:
        breakpoint()
    #res_rmse = torch.pow(output - mu_true, 2).mean().item() # TODO add a square root?
    return mean_mll