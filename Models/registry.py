# -*- coding: utf-8 -*-
"""
Created on Thu Jul 17 09:14:27 2025

@author: NCHAREST
"""

from .hill import HillModel

MODEL_REGISTRY = {
    model.name: model()
    for model in [HillModel] # this list should be updated to register models
    }

def get_model(name: str):
    return MODEL_REGISTRY[name]