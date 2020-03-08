# -*- coding: utf-8 -*-
"""
Created on Sat Jun 16 11:52:25 2018

@author: sofyan.fadli
"""

# Data Preprocessing

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('NEW-DATA-1.T15.csv')

# Number of columns
# print(len(dataset.columns))
# print(dataset.columns)

dataset = dataset.drop(columns=['1:Date'])
dataset = dataset.drop(columns=['2:Time'])
dataset = dataset.drop(columns=['19:Exterior_Entalpic_1'])
dataset = dataset.drop(columns=['20:Exterior_Entalpic_2'])
dataset = dataset.drop(columns=['21:Exterior_Entalpic_turbo'])
dataset = dataset.drop(columns=['24:Day_Of_Week'])

# Number of columns
print(len(dataset.columns))
print(dataset.columns)

X_train = dataset.iloc[:, [0,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17]].values
y_train = dataset.iloc[:, [1]].values

# precipitacion = dataset['19:Exterior_Entalpic_1'].tolist()
# precipitacion_set = set(line for line in precipitacion)

# Taking care of missing data
from sklearn.preprocessing import Imputer
# Kita dpt menggunakan Object Inspector utk melihat parameternya
imputer = Imputer(missing_values= 'NaN', strategy = 'mean', axis = 0)
# Axis mana yang ingin kita perbaiki dengan Object Imputer
imputer = imputer.fit(X_train[:, :])
# Selanjutnya kita pilih axis mana yg ingin kita re-place dgn data yg baru
X_train[:, :] = imputer.transform(X_train[:, :])

# Feature Scaling
from sklearn.preprocessing import StandardScaler

sc_X = StandardScaler()
sc_Y = StandardScaler().fit(y_train)
X_train = sc_X.fit_transform(X_train)
y_train = sc_Y.transform(y_train)

# Fitting Linear Regression to the dataset
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X_train,y_train)

# Fitting Polynomial Regression to the dataset
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 2)
X_poly = poly_reg.fit_transform(X_train)
lin_reg_2 = LinearRegression()
lin_reg_2.fit(X_poly, y_train)

# Importing the data testing
data_train = pd.read_csv('NEW-DATA-2.T15.csv')

data_train = data_train.drop(columns=['1:Date'])
data_train = data_train.drop(columns=['2:Time'])
data_train = data_train.drop(columns=['19:Exterior_Entalpic_1'])
data_train = data_train.drop(columns=['20:Exterior_Entalpic_2'])
data_train = data_train.drop(columns=['21:Exterior_Entalpic_turbo'])
data_train = data_train.drop(columns=['24:Day_Of_Week'])

# Number of columns
print(len(data_train.columns))
print(data_train.columns)

X_test = data_train.iloc[:, [0,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17]].values
y_test = data_train.iloc[:, [1]].values

# Taking care of missing data
from sklearn.preprocessing import Imputer
# Kita dpt menggunakan Object Inspector utk melihat parameternya
imputer = Imputer(missing_values= 'NaN', strategy = 'mean', axis = 0)
# Axis mana yang ingin kita perbaiki dengan Object Imputer
imputer = imputer.fit(X_test[:, :])
# Selanjutnya kita pilih axis mana yg ingin kita re-place dgn data yg baru
X_test[:, :] = imputer.transform(X_test[:, :])

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_test = sc_X.fit_transform(X_test)

# Predicting the Test set Results
y_pred = lin_reg_2.predict(poly_reg.fit_transform(X_test))

y_pred_inverse = sc_Y.inverse_transform(y_pred)
