from typing import Dict

from botorch.acquisition import (
    qLogNoisyExpectedImprovement
)

ACQUISITION_REGISTRY = {
    'lognei': qLogNoisyExpectedImprovement,
}


def parse_acquisition_options(kwargs: Dict) -> Dict:
    return None