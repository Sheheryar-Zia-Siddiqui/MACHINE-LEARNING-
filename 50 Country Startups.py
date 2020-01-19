# -*- coding: utf-8 -*-
"""
Created on Sun Jan 19 00:22:34 2020

@author: Sheheryar Zia Siddiqui
"""


#Importing Libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
# Importing the dataset
dataset = pd.read_csv('50_Startups.csv')


############################   NEW YORK  ############################

# Importing the dataset
p = dataset.loc[dataset['State']=='New York']
print(p)
y = p.iloc[:, -1].values
q =np.arange(17)
X = q.reshape(-1, 1)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20,random_state = 0)

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Fitting Polynomial Regression to the dataset
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 3)
X_poly = poly_reg.fit_transform(X)
poly_reg.fit(X_poly, y)
lin_reg_2 = LinearRegression()
lin_reg_2.fit(X_poly, y)

# Visualising the Polynomial Regression results
plt.scatter(X, y, color = 'red')
plt.plot(X, lin_reg_2.predict(poly_reg.fit_transform(X)), color = 'blue')
plt.title('Startups Profit In New York (Polynomial Regression)')
plt.xlabel('(City: New York)')
plt.ylabel('Profit')
plt.show()

print('Startups Profit In New York (23)')
print(regressor.predict([[23]]))


############################   FLORIDA  ############################

q = dataset.loc[dataset['State']=='Florida']
print(q)
b = q.iloc[:, -1].values
r =np.arange(16)
a = r.reshape(-1, 1)

from sklearn.model_selection import train_test_split
a_train, a_test, b_train, b_test = train_test_split(a, b, test_size = 0.20,random_state = 0)

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(a_train, b_train)

# Fitting Polynomial Regression to the dataset
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 3)
a_poly = poly_reg.fit_transform(a)
poly_reg.fit(a_poly, b)
lin_reg_2 = LinearRegression()
lin_reg_2.fit(a_poly, b)

# Visualising the Polynomial Regression results
plt.scatter(a, b, color = 'red')
plt.plot(a, lin_reg_2.predict(poly_reg.fit_transform(a)), color = 'blue')
plt.title('Startups Profit In Florida (Polynomial Regression)')
plt.xlabel('(City: Florida)')
plt.ylabel('Profit')
plt.show()

print('Startups Profit In Florida (23)')
print(regressor.predict([[23]]))


############################   CALIRORNIA  ############################

s = dataset.loc[dataset['State']=='California']
print(s)
e = s.iloc[:, -1].values
d =np.arange(17)
f = d.reshape(-1, 1)

from sklearn.model_selection import train_test_split
f_train, f_test, e_train, e_test = train_test_split(f, e, test_size = 0.20,random_state = 0)

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(f_train, e_train)

# Fitting Polynomial Regression to the dataset
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 3)
f_poly = poly_reg.fit_transform(f)
poly_reg.fit(f_poly, e)
lin_reg_2 = LinearRegression()
lin_reg_2.fit(f_poly, e)

# Visualising the Polynomial Regression results
plt.scatter(f, e, color = 'red')
plt.plot(f, lin_reg_2.predict(poly_reg.fit_transform(f)), color = 'blue')
plt.title('Startups Profit In California (Polynomial Regression)')
plt.xlabel('(City: California)')
plt.ylabel('Profit')
plt.show()

print('Startups Profit In California (23)')
print(regressor.predict([[23]]))