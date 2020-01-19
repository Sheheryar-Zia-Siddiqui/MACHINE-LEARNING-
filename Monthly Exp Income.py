# -*- coding: utf-8 -*-
"""
Created on Fri Jan 17 22:23:07 2020

@author: Sheheryar Zia Siddiqui
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv('monthlyexpvsincom.csv')
a = dataset.iloc[:, :-1].values
b = dataset.iloc[:, 1].values

from sklearn.linear_model import LinearRegression
# Fitting Polynomial Regression to the dataset
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 4)
a_poly = poly_reg.fit_transform(a)
poly_reg.fit(a_poly, b)
lin_reg_2 = LinearRegression()
lin_reg_2.fit(a_poly, b)

# Visualising the Polynomial Regression results (for higher resolution and smoother curve)
a_grid = np.arange(min(a), max(a), 0.1)
a_grid = a_grid.reshape((len(a_grid), 1))
plt.scatter(a, b, color = 'red')
plt.plot(a_grid, lin_reg_2.predict(poly_reg.fit_transform(a_grid)), color = 'blue')
plt.title('Truth or Bluff (Polynomial Regression)')
plt.xlabel('Monthly Experience')
plt.ylabel('Icome')
plt.show()

# Predicting result with Polynomial Regression
print('The Income Of The Asked Experienced Person Should Be')
print(lin_reg_2.predict(poly_reg.fit_transform([[6.5]])))