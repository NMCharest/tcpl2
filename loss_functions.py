"""Callable enum of available loss functions.

Loss functions have signature (obs, pred, err) -> loss and 
are used in curve-fitting.
"""

from enum import Enum, member, nonmember
from typing import Callable

import numpy as np
from numpy.typing import ArrayLike
from scipy.stats import t, norm


class LossFunctions(Enum):
    """Callable enum of available loss functions."""

    @nonmember
    @staticmethod
    def _loss_fn(
        pdf: Callable[..., ArrayLike],
        **kwargs
    ) -> Callable[[ArrayLike, ArrayLike, ArrayLike], float]:
        """Generic log loss function using any input log PDF function."""
        return lambda o, p, e: np.sum(pdf((o - p) /  e, **kwargs) - np.log(e))

    # t-distributed log error with 4 DoF
    DT4 = member(staticmethod(_loss_fn(t.logpdf, df=4)))
    # Normally distributed log error
    DNORM = member(staticmethod(_loss_fn(norm.logpdf)))

    def __call__(
        self,
        obs: ArrayLike,
        pred: ArrayLike,
        err: ArrayLike
    ) -> float:
        return self.value(obs, pred, err)
