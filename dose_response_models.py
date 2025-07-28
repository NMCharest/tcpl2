"""Library of classes representing various dose-response models.

Classes:
    DoseResponseModel: abstract base class for models
    LogHillModel: Log Hill model ~ f(x) = tp / (1 + 10 ^ (p * (ga - x)))
"""

from abc import ABC, abstractmethod
from typing import Callable, NamedTuple

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike
from scipy.optimize import minimize
from scipy.stats import median_abs_deviation

from loss_functions import LossFunctions

# Default loss function (DT4 or DNORM)
DEFAULT_LOSS_FN = LossFunctions.DT4
# Default bidirectional fit
DEFAULT_BID = True
# Default error (1e-32 or 1e-16)
DEFAULT_ERROR = 1e-32
# Default error bounds
DEFAULT_ERROR_BOUNDS = (-32, 32)
# Default optimization solver
DEFAULT_METHOD = 'Nelder-Mead'


class DoseResponseModel(ABC):
    """Abstract base class defining dose-response model behavior."""

    def __init__(self):
        # Initialize various output parameters
        self.success_ = False
        self.log_likelihood_ = None
        self.aic_ = None
        self.best_fit_ = None
        self.best_params_ = None

    class _Param(NamedTuple):
        name: str
        guess_fn: Callable[[ArrayLike, ArrayLike, bool], float]
        bounds_fn: Callable[[ArrayLike, ArrayLike, bool], tuple[float, float]]

    @property
    @abstractmethod
    def _name(self) -> str:
        """Require model name for derived classes."""

    @property
    @abstractmethod
    def _is_log_fit(self) -> bool:
        """Require flag to fit raw or log x for derived classes."""

    @property
    @abstractmethod
    def _model_params(self) -> list[_Param]:
        """Require list of model parameters for derived classes."""

    @abstractmethod
    def _model_fn(self, tx: ArrayLike, *args) -> ArrayLike:
        """Require model curve-fitting function for derived classes."""

    @staticmethod
    def _error_guess(y: ArrayLike):
        """Initial model-agnostic estimate of error."""
        return (
            np.log(y_mad)
            if (y_mad := median_abs_deviation(y, scale='normal')) > 0
            else np.log(DEFAULT_ERROR)
        )

    @staticmethod
    def _meds(tx: ArrayLike, y: ArrayLike, bid: bool):
        """Calculate median response at each dose if multiple samples."""
        meds = pd.DataFrame({'y': y, 'tx': tx}).groupby('tx')['y'].median()
        return abs(meds) if bid else meds # Absolute value if bidirectional fit

    def _transform_x(self, x: ArrayLike) -> ArrayLike:
        """Calculate log x if model is fitted in log space."""
        return x if not self._is_log_fit else np.log10(x)

    def _transform_error(self, err: float) -> float:
        """Calculate exp error if model is fitted in log space."""
        return err if not self._is_log_fit else np.exp(err)

    def fit(
        self,
        x: ArrayLike,
        y: ArrayLike,
        loss_fn: LossFunctions = DEFAULT_LOSS_FN,
        bid: bool = DEFAULT_BID
    ):
        """Fit the model function to the provided dose-response data.
        
        Args:
            x: untransformed dose data
            y: response data
            loss_fn: loss function (default dt4)
            bid: bidirectional fit (default true)
        Returns:
            fitted model object
        """

        # Perform log transformation of data if needed
        tx = self._transform_x(x)

        # Define objective function
        def obj_fn(params):
            return -loss_fn(
                y,
                self._model_fn(tx, *params[:-1]),
                self._transform_error(params[-1])
            )

        # Guess initial conditions and bounds for model parameters,
        # appending model-agnostic defaults for error parameter
        x0 = [p.guess_fn(tx, y, bid) for p in self._model_params]\
            + [DoseResponseModel._error_guess(y)]
        bounds = [p.bounds_fn(tx, y, bid) for p in self._model_params]\
            + [DEFAULT_ERROR_BOUNDS]

        # Perform optimization
        fit = minimize(
            fun=obj_fn,
            x0=x0,
            bounds=bounds,
            method=DEFAULT_METHOD,
            options={'disp': True}
        )

        # Extract the fit information
        self.best_fit_ = fit
        self.success_ = fit.success
        if fit.success:
            self.best_params_ = fit.x
            self.log_likelihood_ = -fit.fun
            self.aic_ = 2 * (len(self.best_params_) - self.log_likelihood_)

        return self # Permit chaining with predict()

    def predict(self, x: ArrayLike) -> ArrayLike:
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
        bid: bool = DEFAULT_BID
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
    """

    _name = 'loghill'
    _is_log_fit = True

    @staticmethod
    def _tp_bounds_fn(tx, y, bid):
        if bid:
            tp_max = 1.2 * max([abs(min(y)), abs(max(y))])
            tp_min = -tp_max
        else:
            tp_min = 0
            tp_max = 1.2 * max(y)

        return tp_min, tp_max

    _model_params = [
        DoseResponseModel._Param(
            'tp',
            lambda tx, y, bid: DoseResponseModel._meds(tx, y, bid).max(),
            _tp_bounds_fn
        ),
        DoseResponseModel._Param(
            'ga',
            (lambda tx, y, bid:
                DoseResponseModel._meds(tx, y, bid).idxmax() - 0.5),
            lambda tx, y, bid: (min(tx) - 1, max(tx) + 0.5),
        ),
        DoseResponseModel._Param(
            'p',
            lambda tx, y, bid: 1.2,
            lambda tx, y, bid: (0.3, 8)
        )
    ]

    def _model_fn(self, tx, *params):
        return params[0] / (1 + 10 ** (params[2] * (params[1] - tx)))


class Poly1Model(DoseResponseModel):
    """Degree-1 polynomial (linear) model fitting function.
    
    Parameters:
        a: y-scale (slope)
    """

    _name = 'poly1'
    _is_log_fit = False

    @staticmethod
    def _max_slope(tx, y, bid):
        meds = DoseResponseModel._meds(tx, y, bid)
        return meds.max() / max(tx)

    @staticmethod
    def _a_bounds_fn(tx, y, bid):
        val = 1e8 * abs(Poly1Model._max_slope(tx, y, bid))
        return (-val, val) if bid else (0, val)

    _model_params = [DoseResponseModel._Param('a', _max_slope, _a_bounds_fn)]

    def _model_fn(self, tx, *params):
        return params[0] * tx


class PowModel(DoseResponseModel):
    """Power model fitting function.
    
    Parameters:
        a: y-scale
        p: power
    """

    _name = 'pow'
    _is_log_fit = False

    @staticmethod
    def _max_slope(tx, y, bid):
        meds = DoseResponseModel._meds(tx, y, bid)
        return meds.max() / max(tx)

    @staticmethod
    def _a_bounds_fn(tx, y, bid):
        meds_max_abs = abs(DoseResponseModel._meds(tx, y, bid).max())
        val = 1e8 * meds_max_abs
        return (-val, val) if bid else (1e-8 * meds_max_abs, val)

    _model_params = [
        DoseResponseModel._Param(
            'a',
            lambda tx, y, bid: DoseResponseModel._meds(tx, y, bid).max(),
            _a_bounds_fn
        ),
        DoseResponseModel._Param(
            'p',
            lambda tx, y, bid: 1.5,
            lambda tx, y, bid: (-20, 20)
        )
    ]

    def _model_fn(self, tx, *params):
        return params[0] * tx ** params[1]
