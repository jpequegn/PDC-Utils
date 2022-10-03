# AUTOGENERATED! DO NOT EDIT! File to edit: ../nbs/00_PDC.ipynb.

# %% auto 0
__all__ = ['power_curve', 'PDC']

# %% ../nbs/00_PDC.ipynb 4
from lmfit import Model, Parameters
import numpy as np
import pandas as pd

# %% ../nbs/00_PDC.ipynb 6
def power_curve(x, frc, ftp, tte, tau, tau2, a):
    p = frc/x * (1.0 - np.exp(-x/tau)) + ftp * (1 - np.exp(-x / tau2))
    p -= np.maximum(0, a * np.log(x / tte))
    return p

# %% ../nbs/00_PDC.ipynb 7
def PDC():
    def __init__(self, df_x, df_y): self.df_x, self.df_y = df_x, df_y
    
    def fit():
        gmodel = Model(power_curve)
        params = Parameters()
        params.add('frc', value=5000, min=1, max=15000)
        params.add('ftp', value=150, min=100, max=400)
        params.add('tte', value=2000, min=1800, max=3600)
        params.add('tau', value=12, min=10, max=25)
        params.add('tau2', value=5000, min=10, max=25)
        params.add('a', value=10, min=1, max=200)
        
        return gmodel.fit(self.df_y, params, x=self.df_x)
        
    
