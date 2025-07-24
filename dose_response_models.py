"""Library of classes representing various dose-response models.

Classes:
    DoseResponseModel: abstract base class for models
    LogHillModel: Log Hill model ~ f(x) = tp / (1 + 10 ^ (p * (ga - x)))
"""

from abc import ABC, abstractmethod
from typing import NamedTuple

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike
from scipy.optimize import minimize
from scipy.stats import median_abs_deviation

from loss_functions import LossFunctions


class DoseResponseModel(ABC):
    """Abstract base class defining dose-response model behavior."""

    def __init__(self):
        # Initialize various output parameters
        self.success_ = False
        self.log_likelihood_ = None
        self.aic_ = None
        self.best_fit_ = None
        self.best_params_ = None

    DEFAULT_LOSS_FN = LossFunctions.DT4

    @property
    @abstractmethod
    def _name(self) -> str:
        """Require model name for derived classes."""

    @property
    @abstractmethod
    def _fit_log_x(self) -> bool:
        """Require flag to fit raw or log x for derived classes."""

    @abstractmethod
    def _model_fn(
        self,
        tx: ArrayLike,
        *args
    ) -> ArrayLike:
        """Require model curve-fitting function for derived classes."""

    class ParamGuess(NamedTuple):
        """Named tuple representing an initial param guess and bounds."""
        guess: list[float]
        bounds: list[tuple[float, float]]

    @abstractmethod
    def _parameterize_initial(
        self,
        tx: ArrayLike,
        y: ArrayLike,
        bid: bool
    ) -> ParamGuess:
        """Require method to guess initial conditions for derived classes."""

    def _transform_x(self, x: ArrayLike) -> ArrayLike:
        """Calculate log x if model is fitted in log space."""
        return x if not self._fit_log_x else np.log10(x)

    def _transform_err(self, err: float) -> float:
        """Calculate exp err if model is fitted in log space."""
        return err if not self._fit_log_x else np.exp(err)

    def _obj_fn(
        self,
        params: ArrayLike,
        tx: ArrayLike,
        y: ArrayLike,
        loss_fn: LossFunctions
    ) -> float:
        """Compute the objective function used to optimize model fitting.
        
        Args:
            params: model parameters, including error term
            tx: transformed dose data
            y: response data
            loss_fn: loss function
        Returns:
            objective function value
        """
        return -loss_fn(
            y,
            self._model_fn(tx, *params[:-1]),
            self._transform_err(params[-1])
        )

    def fit(
        self,
        x: ArrayLike,
        y: ArrayLike,
        loss_fn: LossFunctions = DEFAULT_LOSS_FN,
        bid: bool = True
    ):
        """Fit the model function to the provided dose-response data.
        
        Args:
            x: untransformed dose data
            y: response data
            loss_fn: loss function (defaults dt4)
            bid: bidirectional fit (default true)
        Returns:
            fitted model object
        """

        # Perform log transformation of data if needed
        tx = self._transform_x(x)
        # Guess initial conditions and bounds
        ic = self._parameterize_initial(tx, y, bid)

        # Perform optimization
        fit = minimize(
            fun=self._obj_fn,
            x0=ic.guess,
            args=(tx, y, loss_fn),
            bounds=ic.bounds,
            method='L-BFGS-B'
        )

        # Extract the fit information
        self.best_fit_ = fit
        self.success_ = fit.success
        if fit.success:
            self.best_params_ = fit.x
            self.log_likelihood_ = -fit.fun
            self.aic_ = 2 * (len(self.best_params_) - self.log_likelihood_)

        return self # Permit chaining with predict()

    def predict(
        self,
        x: ArrayLike
    ) -> ArrayLike:
        """Use fitted model to perform prediction for new dose data.
        
        Args:
            x: new untransformed dose data
        Returns:
            fitted model predictions
        """
        if not self.success_:
            raise ValueError('Model not fit.')
        return self._model_fn(self._transform_x(x), *self.best_params_[:-1])

    def fit_predict(
        self,
        x: ArrayLike,
        y: ArrayLike,
        loss_fn: LossFunctions = DEFAULT_LOSS_FN,
        bid: bool = True
    ) -> ArrayLike:
        """Fit the model then predict from the same data.

        Args:
            x: untransformed dose data
            y: response data
            loss_fn: loss function (default dt4)
            bid: bidirectional fit (default true)
        """
        return self.fit(x, y, loss_fn, bid).predict(x)


class LogHillModel(DoseResponseModel):
    """Hill model fitting function in log space.
    
    Parameters:
        tp: theoretical maximal response (top)
        ga: gain AC50
        p: gain power
        er: error term
    """

    _name = 'loghill'
    _fit_log_x = True

    def _model_fn(self, tx, *args):
        return args[0] / (1 + 10 ** (args[2] * (args[1] - tx)))

    def _parameterize_initial(self, tx, y, bid):
        # Calculate median response at each dose in case of multiple samples
        meds = pd.DataFrame({'y': y, 'tx': tx}).groupby('tx')['y'].median()
        # Initial parameter guesses
        guess = [
            abs(meds).max() if bid else meds.max(), # tp0
            meds.idxmax() - 0.5, # ga0
            1.2, # p0
            (np.log(y_mad) if (y_mad := median_abs_deviation(y)) > 0
                else np.log(1e-32)) # er0
        ]

        # Bounds for tp depend on whether fit is bidirectional
        if bid:
            tp_max = 1.2 * max([abs(min(y)), abs(max(y))])
            tp_min = -tp_max
        else:
            tp_min = 0
            tp_max = 1.2 * max(y)

        bounds = [
            (tp_min, tp_max), # tp
            (min(tx) - 1, max(tx) + 0.5), # ga
            (0.3, 8), # p
            (-20, 5), # er
        ]

        return DoseResponseModel.ParamGuess(guess=guess, bounds=bounds)
