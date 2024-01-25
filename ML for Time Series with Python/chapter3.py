#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 24 11:51:43 2024

@author: ratsimbazafy
"""


from scipy.optimize import minimize
import numpy as np
np.random.seed(0)
pts = 10000
vals = np.random.lognormal(0, 1.0, pts)

from sklearn.preprocessing import StandardScaler
from scipy.stats import normaltest
scaler = StandardScaler()
vals_ss = scaler.fit_transform(vals.reshape(-1,1))
_, p = normaltest(vals_ss)
# print(f'significance: {p:.2f}')

log_transformed = np.log(vals)
_, p = normaltest(log_transformed)
print(p)

from scipy.stats import boxcox 
vals_bc = boxcox(vals, 0.0)

# Imputation

from sklearn.impute import SimpleImputer
imp_mean = SimpleImputer(missing_values=np.nan, strategy='mean')
imp_mean.fit([[7,2,3], [4, np.nan, 6], [10,5,9]])
SimpleImputer()
df = [[np.nan, 2, 3], [4,np.nan,6], [10, np.nan, 9]]
print(imp_mean.transform(df))

# Holiday Features

from workalendar.europe.united_kingdom import UnitedKingdom
UnitedKingdom().holidays()