from typing import Dict

from botorch.acquisition import (
    qExpectedImprovement,
)
from botorch.acquisition.logei import (
    qLogNoisyExpectedImprovement,
)
from botorch.acquisition.multi_objective import (
    qNoisyExpectedHypervolumeImprovement,
)
from botorch.acquisition.bayesian_active_learning import (
    BayesianActiveLearningByDisagreement,
    StatisticalDistanceActiveLearning,
)
from botorch.acquisition.scorebo import (
    SelfCorrectingBayesianOptimization
)

ACQUISITION_REGISTRY = {
    'LogNEI': qLogNoisyExpectedImprovement,
    'NEHVI': qNoisyExpectedHypervolumeImprovement,
    'EI': qExpectedImprovement,
    'BALD': BayesianActiveLearningByDisagreement,
    'SAL': StatisticalDistanceActiveLearning,
    'SCoreBO': SelfCorrectingBayesianOptimization,

}


def parse_acquisition_options(kwargs: Dict) -> Dict:
    return {}