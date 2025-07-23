# -*- coding: utf-8 -*-
"""
Created on Thu Jul 17 08:45:31 2025

@author: NCHAREST
"""

from abc import ABC, abstractmethod
import numpy as np
from scipy.optimize import minimize

class DoseResponseModel(ABC):
    name: str
    def __init__(self, loss_fn = None):
        self.loss_fn = loss_fn
        self.params_ = None
        self.success_ = False
        self.fit_result_ = None
    
    @abstractmethod
    def model_function(self, logc, *params):
        pass
    
    @abstractmethod
    def initial_guess(self, conc, y_obs):
        """initial parameter guesses"""
        pass
    
    def loss(self, params, conc, y_obs):
        logc = np.log10(conc)
        *core_params, log_err = params
        err = np.exp(log_err)
        y_pred = self.model_function(logc, *core_params)
        return -self.loss_fn(y_obs, y_pred, err)

    def fit(self, conc, resp):
        logc = np.log10(conc)
        guess, bounds = self.initial_guess(conc, resp)
        result = minimize(
            fun=self.loss,
            x0 = guess,
            args=(conc, resp),
            bounds=bounds,
            method="L-BFGS-B"
            )
        
        self.fit_result_ = result
        self.success_ = result.success
        if result.success: 
            self.params_ = result.x 
            *core_params, log_err = self.params_
            y_pred = self.model_function(logc, *core_params)
            self.log_likelihood_ = self.loss_fn(y_obs=resp, y_pred=y_pred, err=np.exp(log_err))
            self.aic_ = 2*len(self.params_) - 2 * self.log_likelihood_ # compute fit AIC
        return self
    
    def predict(self, conc):
        if not self.success_:
            raise ValueError("Model not fit.")    
        logc = np.log10(conc)
        return self.model_function(logc, *self.params_[:-1])