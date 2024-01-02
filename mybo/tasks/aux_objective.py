from typing import Callable
import torch
from torch import Tensor
from torch.quasirandom import SobolEngine
from botorch.utils.transforms import unnormalize
from botorch.test_functions import SyntheticTestFunction
from gpytorch.mlls import ExactMarginalLogLikelihood
from ax.service.ax_client import AxClient

def evaluate_mll(ax_client: AxClient, objective: SyntheticTestFunction, num_test_points: int):
    try:
        model = ax_client.get_model_predictions()
    except NotImplementedError:
        return -1e6
    sobol = SobolEngine(len(objective.bounds), scramble=True, seed=42)
    test_batch = sobol.draw(num_test_points)
    gp = ax_client.generation_strategy.model.model.surrogate.model
    output = - \
        objective.evaluate_true(unnormalize(test_batch, objective.bounds))
    y_transform = ax_client._generation_strategy.model.transforms['StandardizeY']
    objective_name = list(ax_client.experiment.metrics.keys())[0]
    y_mean, y_std = y_transform.Ymean[objective_name], y_transform.Ystd[objective_name]
    mu, _ = ax_client._generation_strategy.model.model.predict(
                test_batch)
    mu_true = (mu * y_std + y_mean).flatten()

    model = ax_client._generation_strategy.model.model.surrogate.model
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    model.eval()
    preds = model(test_batch)
    norm_yvalid = (output - y_mean) / y_std

    norm_yvalid = norm_yvalid.flatten()
    # marg_dist = MultivariateNormal(predmean, predcov)
    # joint_loglik = -mll(marg_dist, norm_yvalid).mean()
    res_mll = mll(preds, norm_yvalid).mean().item()
    res_rmse = torch.pow(output - mu_true, 2).mean().item() # TODO add a square root?
    return res_mll