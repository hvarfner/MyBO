import torch
from torch import Tensor

from gpytorch.kernels import (
    ScaleKernel,
    RBFKernel,
    MaternKernel,
)
from gpytorch.models import ExactGP
from gpytorch.means import ConstantMean
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.likelihoods import (
    GaussianLikelihood, 
    GaussianLikelihoodWithMissingObs
)
from gpytorch.priors import GammaPrior, LogNormalPrior, NormalPrior
from gpytorch.distributions import MultivariateNormal
import botorch
from botorch.fit import fit_gpytorch_mll as fit
botorch.settings.debug(True)
import matplotlib.pyplot as plt


add_nans = False
train_X = torch.Tensor([
    [0.1],
    [0.3],
    [0.5],
    [0.7],
    [0.9],
])
train_Y = torch.sin(7 * train_X)

missing_X = torch.linspace(0, 1, 31).unsqueeze(-1)
missing_Y = torch.zeros_like(missing_X) * torch.nan

if add_nans:
    likelihood = GaussianLikelihoodWithMissingObs(noise_prior=GammaPrior(2.0, 0.15))
    train_X = torch.cat((train_X, missing_X))
    train_Y = torch.cat((train_Y, missing_Y))

else:
    likelihood = GaussianLikelihood(noise_prior=GammaPrior(2.0, 0.15))

class MyGP(ExactGP):
    def __init__(self, train_inputs, train_targets, likelihood):
        super().__init__(
            train_inputs=train_inputs, 
            train_targets=train_targets, 
            likelihood=likelihood
        )
        self.mean_module = ConstantMean(prior=NormalPrior(0, 1))
        self.covar_module = ScaleKernel(MaternKernel(lengthscale_prior=GammaPrior(3.0, 6.0)))
        self.covar_module = ScaleKernel(MaternKernel(lengthscale_prior=GammaPrior(3.0, 6.0)))
    
    def forward(self, x):
        mean = self.mean_module(x)
        covar = self.covar_module(x)
        bre
        return MultivariateNormal(mean, covar)
    
    def posterior(self, X):
        self.eval()
        mvn = self(X)
        return mvn

gp = MyGP(
    train_inputs=train_X, 
    train_targets=train_Y, 
    likelihood=likelihood,
)

mll = ExactMarginalLogLikelihood(likelihood, gp)
fit(mll)

X_plot = torch.linspace(0, 1, 101)
post = gp.posterior(X_plot.unsqueeze(-1))
with torch.no_grad():
    mean = post.mean.numpy().flatten()
    std = 2 * post.variance.sqrt().numpy().flatten()

    plt.plot(X_plot, mean, color="blue", alpha=1)
    plt.fill_between(X_plot, mean - std, mean + std, color="blue", alpha=0.15)
    plt.scatter(train_X.flatten(), train_Y.flatten(), s=50, color="k")

plt.show()
