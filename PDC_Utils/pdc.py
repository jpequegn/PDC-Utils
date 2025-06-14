"""Utilities to manipulate power duration curves, fit them and do what-if analysis"""

# AUTOGENERATED! DO NOT EDIT! File to edit: ../nbs/01_PDC.ipynb.

# %% auto 0
__all__ = ['power_curve', 'PDC']

# %% ../nbs/01_PDC.ipynb 4
from lmfit import Model, Parameters
import numpy as np
import pandas as pd

# %% ../nbs/01_PDC.ipynb 6
def power_curve(x, 
                frc,  # Functional Reserve Capacity 
                ftp,  # Functional Threshold Power
                tte,  # Time to Exhaustion
                tau,  # Short end calibration
                tau2, # Long end calibration
                a): # Decay factor past TTE
    p = frc/x * (1.0 - np.exp(-x/tau)) + ftp * (1 - np.exp(-x / tau2))
    p -= np.maximum(0, a * np.log(x / tte))
    return p

# %% ../nbs/01_PDC.ipynb 7
class PDC:
    "A Power Duraction Curve"
    def __init__(self, x, y): self.x, self.y = x, y
    
    def fit(self):
        gmodel = Model(power_curve)
        params = Parameters()
        params.add('frc', value=5000, min=1, max=15000)
        params.add('ftp', value=150, min=100, max=400)
        params.add('tte', value=2000, min=1800, max=3600)
        params.add('tau', value=12, min=10, max=25)
        params.add('tau2', value=5000, min=10, max=25)
        params.add('a', value=10, min=1, max=200)
        
        return gmodel.fit(self.y, params, x=self.x)
        
    
