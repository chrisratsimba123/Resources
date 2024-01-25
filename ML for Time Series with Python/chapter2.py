#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 24 10:28:50 2024

@author: ratsimbazafy
"""

import datetime
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

pollution = pd.read_csv(
    'https://raw.githubusercontent.com/owid/owid-datasets/master/datasets/Air%20pollution%20by%20city%20-%20Fouquet%20and%20DPCC%20(2011)/Air%20pollution%20by%20city%20-%20Fouquet%20and%20DPCC%20(2011).csv'
)

print(pollution.shape)
print(pollution.columns)

pollution = pollution.rename(
    columns={
        'Smoke (Fouquet and DPCC (2011))': 'Smoke',
        'Suspended Particulate Matter (SPM) (Fouquet and DPCC (2011))': 'SPM',
        'Entity': 'City'
        }
    )

print(pollution.dtypes)

print(pollution['City'].unique())
print(pollution['Year'].min(), pollution['Year'].max())

pollution['Year'] = pollution['Year'].apply(
    lambda x: datetime.datetime.strptime(str(x), '%Y')
    )
print(pollution.dtypes)

print(pollution.isnull().mean())
print(pollution.describe())

n, bins, patches = plt.hist(
    x=pollution['SPM'], bins='auto',
    alpha=0.7, rwidth=0.85
    )

plt.grid(axis='y', alpha=0.75)
plt.xlabel('SPM')
plt.ylabel('Frequency')
plt.show()

pollution = pollution.pivot('Year', 'City', 'SPM')
plt.figure(figsize=(12,6))
sns.lineplot(data=pollution)
plt.ylabel('SPM')

temps = pd.read_csv('monthly_csv.csv')
temps['Date'] = pd.to_datetime(temps['Date'])
temps = temps.pivot('Date', 'Source', 'Mean')

# Fit Seasonal Variation and Trend
from numpy import polyfit

def fit(X, y, degree=3):
    coef = polyfit(X, y, degree)
    trendpoly = np.poly1d(coef)
    return trendpoly(X)

def get_season(s, yearly_periods=4, degree=3):
    X = [i%(365/4) for i in range(0, len(s))]
    seasonal = fit(X, s.values, degree)
    return pd.Series(data=seasonal, index=s.index)

def get_trend(s, degree=3):
    X = list(range(len(s)))
    trend = fit(X, s.values, degree)
    return pd.Series(data=trend, index=s.index)

plt.figure(figsize=(12,6))
temps['trend'] = get_trend(temps['GCAG'])
temps['season'] = get_season(temps['GCAG'] - temps['trend'])
sns.lineplot(data=temps[['GCAG', 'season', 'trend']])
plt.ylabel('Temperature change')

pollution = pollution.pivot('Year', 'City', 'SPM')
pd.plotting.autocorrelation_plot(pollution['London'])

from statsmodels.tsa import stattools
stattools.adfuller(pollution['London'])

































