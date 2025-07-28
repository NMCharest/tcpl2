"""Library of dose-response model classes for power models.

Classes:
    PowerModel: Power model ~ f(x) = a * x ^ p
"""

from numpy.typing import ArrayLike

from .base import DoseResponseModel


class PowerModel(DoseResponseModel):
    """Power model fitting function.
    
    Parameters:
        a: y-scale
        p: power
    """

    _name = 'pow'
    _is_log_fit = False

    @staticmethod
    def _a_bounds_fn(
        tx: ArrayLike,
        y: ArrayLike,
        bid: bool
    ) -> tuple[float, float]:
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

    def _model_fn(self, tx: ArrayLike, *params):
        return params[0] * tx ** params[1]
