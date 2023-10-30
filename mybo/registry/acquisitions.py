from typing import Dict

from botorch.acquisition.logei import (
    qLogNoisyExpectedImprovement,
)
from botorch.acquisition.multi_objective import (
    qNoisyExpectedHypervolumeImprovement,
)

ACQUISITION_REGISTRY = {
    'lognei': qLogNoisyExpectedImprovement,
    'nehvi': qNoisyExpectedHypervolumeImprovement,
}


def parse_acquisition_options(kwargs: Dict) -> Dict:
    return {}