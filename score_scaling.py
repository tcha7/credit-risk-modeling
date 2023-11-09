import numpy as np
import pandas as pd
from typing import Union


def scale_score(
    score: Union[np.ndarray, pd.Series, pd.DataFrame],
    pdo: float = 60,
    base_odd: float = 1,
    base_score: float = 600,
) -> Union[np.ndarray, pd.Series, pd.DataFrame]:
    """
    odds = prob_good/prob_bad
    PDO = points-to-double-odds = the number of points before the odd doubles
    base_odd and base_score are the pre-defined point. For example, if we want to define
    the score 600 to have odd 1, then we set base_odd = 1, and base_score = 600
    """

    alpha = pdo / np.log(2)
    scaled_score = (
        base_score - alpha * np.log(base_odd) + alpha * np.log((1 - score) / score)
    )
    return scaled_score
