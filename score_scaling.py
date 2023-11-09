import numpy as np
import pandas as pd
from typing import Union


def log_pdo_scale(
    score: Union[np.ndarray, pd.Series, pd.DataFrame],
    pdo: float = 60,
    base_odd: float = 1,
    base_score: float = 600,
) -> Union[np.ndarray, pd.Series, pd.DataFrame]:
    """
    score = probability of bad of which value is in [0,1], where 1 is 100% bad.
    odds = prob_good/prob_bad
    PDO = points-to-double-odds = the number of points before the odd doubles
    base_odd and base_score are the pre-defined point. For example, if we want to define
    the score 600 to have odd 1, then we set base_odd = 1, and base_score = 600
    """

    _check_input_score(score)

    alpha = pdo / np.log(2)
    scaled_score = (
        base_score - alpha * np.log(base_odd) + alpha * np.log((1 - score) / score)
    )

    return scaled_score


def _check_input_score(score: Union[np.ndarray, pd.Series, pd.DataFrame]) -> None:
    if not _correct_input_score_type(score):
        raise Exception(
            "`score` should be a numpy array, pandas series, or pandas dataframe."
        )

    if not _correct_input_score_range(score):
        raise Exception("`score` should be in the range of [0,1] with 1 being high PD.")


def _correct_input_score_range(
    score: Union[np.ndarray, pd.Series, pd.DataFrame]
) -> bool:
    if (score >= 0).all() & (score <= 1).all():
        return True
    else:
        return False


def _correct_input_score_type(
    score: Union[np.ndarray, pd.Series, pd.DataFrame]
) -> bool:
    if isinstance(score, (np.ndarray, pd.Series, pd.DataFrame)):
        return True
    else:
        return False
