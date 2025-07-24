"""Library of classes representing various dose-response models.

Dose-response models are used in curve-fitting.

Classes:
    DoseResponseModel: abstract base class for models
    HillModel: Hill model ~ f(x) = tp / (1 + (ga / x) ^ p)
"""

from abc import ABC, abstractmethod
from typing import NamedTuple

import numpy as np
from numpy.typing import ArrayLike
from scipy.optimize import minimize

from loss_functions import LossFunctions


class DoseResponseModel(ABC):
    """Abstract base class defining dose-response model behavior."""

    def __init__(self, loss_fn: LossFunctions):
        self.loss_fn = loss_fn
        self.success_ = False
        self.log_likelihood_ = None
        self.aic_ = None
        self.best_fit_ = None
        self.best_params_ = None

    @property
    @abstractmethod
    def _name(self) -> str:
        """Require a defined private model name for derived classes."""

    @property
    @abstractmethod
    def _fit_log_conc(self) -> bool:
        """Require a private flag for raw or log conc for derived classes."""

    def _transform_conc(self, conc: ArrayLike) -> ArrayLike:
        """Calculate log conc if model is set to fit in log space."""
        return conc if not self._fit_log_conc else np.log10(conc)

    def _transform_err(self, err: float) -> float:
        """Calculate log err if model is set to fit in log space."""
        return err if not self._fit_log_conc else np.exp(err)

    @abstractmethod
    def _fit_fn(
        self,
        tconc: ArrayLike,
        *args
    ) -> ArrayLike:
        """Require a model curve-fitting function for derived classes."""

    class ParamGuess(NamedTuple):
        """Named tuple representing an initial param guess and bounds."""
        guess: list[float]
        bounds: list[tuple[float, float]]

    @abstractmethod
    def _param_guess(
        self,
        tconc: ArrayLike,
        resp: ArrayLike
    ) -> ParamGuess:
        """Require a _guess method for initial params for derived classes."""

    def _loss(self, params, tconc, resp):
        *core_params, err = params
        pred = self._fit_fn(tconc, *core_params)
        return -self.loss_fn(resp, pred, self._transform_err(err))

    def fit(self, conc, resp):
        tconc = self._transform_conc(conc)
        ic = self._param_guess(tconc, resp)
        fit = minimize(
            fun=self._loss,
            x0=ic.guess,
            args=(tconc, resp),
            bounds=ic.bounds,
            method='L-BFGS-B'
        )

        self.best_fit_ = fit
        self.success_ = fit.success
        if fit.success:
            self.best_params_ = fit.x
            *core_params, err = self.best_params_
            terr = self._transform_err(err)
            pred = self._fit_fn(tconc, *core_params)
            self.log_likelihood_ = self.loss_fn(resp, pred, terr)
            self.aic_ = 2 * len(self.best_params_) - 2 * self.log_likelihood_

    def predict(self, conc):
        if not self.success_:
            raise ValueError('Model not fit.')

        return self._fit_fn(
            self._transform_conc(conc), *self.best_params_[:-1])


class HillModel(DoseResponseModel):

    _name = 'hill'
    _fit_log_conc = True

    def _fit_fn(self, tconc, *args):
        return args[0] / (1 + 10 ** (args[2] * (args[1] - tconc)))

    def _param_guess(self, tconc, resp):
        tp0 = np.max(resp)
        ga0 = np.median(tconc)
        p0 = 1.0
        log_err0 = np.log(np.std(resp) + 1e-6)
        guess = [tp0, ga0, p0, log_err0]
        bounds = [
            (0, np.max(resp)), # tp
            (-3, 3), # ga
            (0.1, 10), # p
            (-20, 5), # log_err
        ]

        return DoseResponseModel.ParamGuess(guess=guess, bounds=bounds)
