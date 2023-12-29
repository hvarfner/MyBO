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
    qBayesianActiveLearningByDisagreement,
    qStatisticalDistanceActiveLearning,
)
from botorch.acquisition.scorebo import (
    qSelfCorrectingBayesianOptimization
)

ACQUISITION_REGISTRY = {
    'LogNEI': qLogNoisyExpectedImprovement,
    'NEHVI': qNoisyExpectedHypervolumeImprovement,
    'EI': qExpectedImprovement,
    'BALD': qBayesianActiveLearningByDisagreement,
    'SAL': qStatisticalDistanceActiveLearning,
    'SCoreBO': qSelfCorrectingBayesianOptimization,

}


def parse_acquisition_options(kwargs: Dict) -> Dict:
    return {}