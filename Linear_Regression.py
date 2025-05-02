# -*- coding: utf-8 -*-
"""
Created on Sun Apr 27 18:07:07 2025

@author: Sampati Anvekar
"""

#IMPORTING THE LIBRARIES
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#READING THE DATA FROM FILES
data = pd.read_csv('advertising.csv')
print(data.head())


#TO VISUALIZE DATA
fig, axs = plt.subplots(1,3,sharey = True)
data.plot(kind='scatter',x='TV',y='Sales',ax=axs[0],figsize=(14,7))
data.plot(kind='scatter',x='Radio',y='Sales',ax=axs[1],figsize=(14,7))
data.plot(kind='scatter',x='Newspaper',y='Sales',ax=axs[2],figsize=(14,7))


#CREATING X&Y FOR LINEAR REGRESSION
feature_cols = ['TV']
X = data[feature_cols]
Y = data.Sales

#IMPORTING LINEAR REGRESSION ALGO
from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(X,Y)

print(lr.intercept_)
print(lr.coef_)

result=6.97+0.0554*50
print(result)

X_new = pd.DataFrame({'TV':[data.TV.min(),data.TV.max()]})
print(X_new.head())

preds = lr.predict(X_new)
print(preds)

data.plot(kind='scatter',x='TV',y='Sales')

plt.plot(X_new['TV'],preds,c='red',linewidth=3)

import statsmodels.formula.api as smf
lm=smf.ols(formula = 'Sales ~ TV',data = data).fit()
print(lm.conf_int())

#FINDING THE PROBABILITY VALUE
print(lm.pvalues)


#FINDING THE R-SQUARED VALUES
print(lm.rsquared)

#MULTI LINEAR REGRESSION
print('MULTI LINEAR REGRESSION')
feature_cols = ['TV','Radio','Newspaper']
X = data[feature_cols]
Y = data.Sales

lr = LinearRegression()
lr.fit(X,Y)

print(lr.intercept_)
print(lr.coef_)

lm=smf.ols(formula = 'Sales ~ TV+Radio+Newspaper',data = data).fit()
print(lm.conf_int())

print('SUMMMARY')
print(lm.summary())

lm=smf.ols(formula = 'Sales ~ TV+Radio',data = data).fit()
print(lm.conf_int())

print('SUMMMARY-2')
print(lm.summary())