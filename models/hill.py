"""Library of dose-response model classes for Hill-type models.

Classes:
    LogHillModel: Log Hill model ~ f(x) = tp / (1 + 10 ^ (p * (ga - x)))
"""

from numpy.typing import ArrayLike

from .base import DoseResponseModel


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
    def _tp_bounds_fn(
        tx: ArrayLike,
        y: ArrayLike,
        bid: bool
    ) -> tuple[float, float]:
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
            lambda tx, y, bid: (min(tx) - 1, max(tx) + 0.5)
        ),
        DoseResponseModel._Param(
            'p',
            lambda tx, y, bid: 1.2,
            lambda tx, y, bid: (0.3, 8)
        )
    ]

    def _model_fn(self, tx: ArrayLike, *params):
        return params[0] / (1 + 10 ** (params[2] * (params[1] - tx)))


class LogGainLossModel(DoseResponseModel):
    """Hill gain-loss model fitting function in log space.
    
    Parameters:
        tp: theoretical maximal response (top)
        ga: gain AC50
        p: gain power
        la: loss AC50
        q: loss power
    """

    _name = 'loggnls'
    _is_log_fit = True

    def __init__(self, minwidth):
        super().__init__()
        self.minwidth = minwidth

    @staticmethod
    def _tp_bounds_fn(
        tx: ArrayLike,
        y: ArrayLike,
        bid: bool
    ) -> tuple[float, float]:
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
            lambda tx, y, bid: (min(tx) - 1, max(tx) + 0.5)
        ),
        DoseResponseModel._Param(
            'p',
            lambda tx, y, bid: 1.2,
            lambda tx, y, bid: (0.3, 8)
        ),
        DoseResponseModel._Param(
            'la',
            (lambda tx, y, bid:
                DoseResponseModel._meds(tx, y, bid).idxmax() + 1.01),
            lambda tx, y, bid: (min(tx) - 1, max(tx) + 2)
        ),
        DoseResponseModel._Param(
            'q',
            lambda tx, y, bid: 5,
            lambda tx, y, bid: (0.3, 8)
        )
    ]

    # TODO ga-la min width constraint

    def _model_fn(self, tx: ArrayLike, *params):
        gn = 1 / (1 + 10^((params[1] - tx) * params[2]))
        ls = 1 / (1 + 10^((tx - params[3]) * params[4]))
        return params[0] * gn * ls
