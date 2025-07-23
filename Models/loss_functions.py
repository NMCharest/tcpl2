# -*- coding: utf-8 -*-
"""
Created on Mon Jul 21 10:32:22 2025

@author: NCHAREST
"""

from abc import ABC, abstractmethod
import numpy as np
from scipy.stats import t, norm

class LossFunction(ABC):
    @abstractmethod
    def __call__(self, y_obs, y_pred, err):
        pass

# log[p(y | mu, sigma)] under t-distribution
class T4Loss(LossFunction):
    def __call__(self, y_obs, y_pred, err):
        z = (y_obs - y_pred) / err
        logpdf = t.logpdf(z, df=4)
        return np.sum(logpdf - np.log(err))

# log[p(y | mu, sigma)] under normal distribution
class NormLogLoss(LossFunction):
    def __call__(self, y_obs, y_pred, err):
        z = (y_obs - y_pred) / err
        logpdf = norm.logpdf(z)
        return np.sum(logpdf - np.log(err))
