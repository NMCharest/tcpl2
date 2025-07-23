# -*- coding: utf-8 -*-
"""
Created on Thu Jul 17 09:09:58 2025

@author: NCHAREST
"""

import numpy as np
from base import DoseResponseModel

class HillModel(DoseResponseModel):
    name = "hill"
    """
    Hill Model
    Parameters:
        tp - top, thoeretical maximal response
        ga - gain AC50
        p - gain power
    
    """
    def model_function(self, logc, tp, ga, p):
        return tp / (1 + 10**(p*(ga - logc)))
    
    def initial_guess(self, conc, resp):
        logc = np.log10(conc)
        tp0 = np.max(resp)
        ga0 = np.median(logc)
        p0 = 1.0
        log_err0 = np.log(np.std(resp) + 1e-6)
        guess = [tp0, ga0, p0, log_err0] # note parameters are being fitted in log10 space - need to convert back for values like AC50 in real space
        bounds = [
                 (0, np.max(resp)), # tp
                 (-3, 3), # ga
                 (0.1, 10), # p
                 (-20, 5), # log_err
                  ]
        return guess, bounds

        