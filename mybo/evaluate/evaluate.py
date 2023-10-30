from typing import Any, List, Dict
import numpy as np


def mock_evaluate_mo(metric_names: List[str]) -> Dict[str, float]:
    res = {mn: np.random.uniform(size=1)[0] for mn in metric_names.split()}
    return res