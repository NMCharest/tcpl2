"""Library of classes representing various dose-response models.

Classes:
    DoseResponseModel: abstract base class for models
    LogHillModel: Log Hill model ~ f(x) = tp / (1 + 10 ^ (p * (ga - x)))
"""

from abc import ABC, abstractmethod
from typing import NamedTuple

import numpy as np
from numpy.typing import ArrayLike
from scipy.optimize import minimize

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
        y: ArrayLike
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
        loss_fn: LossFunctions = LossFunctions.DT4
    ):
        """Fit the model function to the provided dose-response data.
        
        Args:
            x: untransformed dose data
            y: response data
            loss_fn: loss function (defaults to dt4)
        Returns:
            fitted model object
        """

        # Perform log transformation of data if needed
        tx = self._transform_x(x)
        # Guess initial conditions and bounds
        ic = self._parameterize_initial(tx, y)

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
        loss_fn: LossFunctions
    ) -> ArrayLike:
        """Fit the model then predict from the same data."""
        return self.fit(x, y, loss_fn).predict(x)


class LogHillModel(DoseResponseModel):
    """Hill model fitting function in log space."""

    _name = 'loghill'
    _fit_log_x = True

    def _model_fn(self, tx, *args):
        return args[0] / (1 + 10 ** (args[2] * (args[1] - tx)))

    def _parameterize_initial(self, tx, y):
        tp0 = np.max(y)
        ga0 = np.median(tx)
        p0 = 1.0
        log_err0 = np.log(np.std(y) + 1e-6)
        guess = [tp0, ga0, p0, log_err0]
        bounds = [
            (0, np.max(y)), # tp
            (-3, 3), # ga
            (0.1, 10), # p
            (-20, 5), # log_err
        ]

        return DoseResponseModel.ParamGuess(guess=guess, bounds=bounds)
