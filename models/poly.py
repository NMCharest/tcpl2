"""Library of dose-response model classes for polynomial models.

Classes:
    LinearModel: Linear model ~ f(x) = a * x
"""

from numpy.typing import ArrayLike

from .base import DoseResponseModel


class LinearModel(DoseResponseModel):
    """Linear (polynomial degree 1) model fitting function.
    
    Parameters:
        a: slope (y-scale)
    """

    _name = 'poly1'
    _is_log_fit = False

    @staticmethod
    def _max_slope(tx: ArrayLike, y: ArrayLike, bid: bool) -> float:
        meds = DoseResponseModel._meds(tx, y, bid)
        return meds.max() / max(tx)

    @staticmethod
    def _a_bounds_fn(
        tx: ArrayLike,
        y: ArrayLike,
        bid: bool
    ) -> tuple[float, float]:
        val = 1e8 * abs(LinearModel._max_slope(tx, y, bid))
        return (-val, val) if bid else (0, val)

    _model_params = [DoseResponseModel._Param('a', _max_slope, _a_bounds_fn)]

    def _model_fn(self, tx: ArrayLike, *params):
        return params[0] * tx
