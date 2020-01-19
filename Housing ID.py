# -*- coding: utf-8 -*-
"""
Created on Tue Jan 14 15:52:56 2020

@author: Sheheryar Zia Siddiqui
"""
import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv('housing price.csv')
X = dataset.iloc[:, :-1 ].values
y = dataset.iloc[:, 1].values

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20,random_state = 0)

from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(X_train, y_train)

plt.scatter(X_train,y_train,color='black')
plt.plot(X_train,regressor.predict(X_train),color='blue')
plt.title('Housing Price As Per ID(Training Set)')
plt.xlabel('House ID')
plt.ylabel('Price Of House')
plt.show()

plt.scatter(X_test,y_test,color='black')
plt.plot(X_train,regressor.predict(X_train),color='blue')
plt.title('Housing Price As Per ID(Test Set)')
plt.xlabel('House ID')
plt.ylabel('Price Of House')
plt.show()

print('Price Of Your Concerned ID House Is')
print(regressor.predict([[3000]]))