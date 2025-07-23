# -*- coding: utf-8 -*-
"""
Created on Wed Jul 23 08:37:01 2025

@author: NCHAREST
"""

import numpy as np
from scipy.optimize import minimize
from hill import HillModel
from scipy.stats import t
import matplotlib.pyplot as plt

conc = np.array([0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0])
resp = np.array([0, 0, 0.1, 0.2, 0.5, 1.0, 1.5, 2.0])

#%%
from loss_functions import T4Loss, NormLogLoss
import matplotlib.pyplot as plt

model = HillModel(loss_fn=T4Loss())

model.fit(conc, resp)
print("Fitting successful:", model.success_)
print("Fitted parameters:", model.params_)

if model.success_:
    print("Parameters:", model.params_)
    print("Predictions:", model.predict(conc))
    logc = np.log10(conc)
    logc_fine = np.linspace(np.min(logc), np.max(logc), 200)
    conc_fine = 10**logc_fine
    pred = model.predict(conc_fine)
    
    fig, ax = plt.subplots()
    ax.scatter(conc, resp, label="Observed", color="black")
    ax.plot(conc_fine, pred, label="Fit", color="blue")
    ax.axvline(10**model.params_[1], linestyle="--", c="red")
    plt.show()
else:
    print("Fit failed.")
        