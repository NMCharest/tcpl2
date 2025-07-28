"""TCPL scratch testing notebook."""

import numpy as np
import matplotlib.pyplot as plt

from models.hill import LogHillModel
from models.poly import LinearModel
from models.power import PowerModel

# Mock data
conc = np.array([0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0])
hill_resp = np.array([0, 0, .1, .2, .5, 1, 1.5, 2])
poly1_resp = np.array([0, .01, .1, .1, .2, .5, 2, 5])
pow_resp = np.array([0, .01, .1, .1, .2, .5, 2, 8])

# Initialize and fit the model
model = PowerModel()
model.fit(conc, pow_resp)

if model.success_:
    # Output results
    print('Fit successful!')
    print(f'Parameters: {model.best_params_}')
    print(f'Log-likelihood: {model.log_likelihood_}')
    print(f'AIC: {model.aic_}')
    print(f'Predictions: {model.predict(conc)}')

    # Predict spaced data for plotting
    logc = np.log10(conc)
    logc_fine = np.linspace(np.min(logc), np.max(logc), 200)
    conc_fine = 10 ** logc_fine
    pred = model.predict(conc_fine)

    # Generate plot
    fig, ax = plt.subplots()
    ax.scatter(conc, pow_resp, label='Observed', color='black')
    ax.plot(conc_fine, pred, label='Fit', color='blue')
    # Add AC50 line
    ax.axvline(10 ** model.best_params_[1], linestyle='--', c='red')
    plt.show()
else:
    print('Fit failed.')
