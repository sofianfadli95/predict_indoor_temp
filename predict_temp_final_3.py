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

X_train = dataset.iloc[:, [0]].values
y_train = dataset.iloc[:, 1].values

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


from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C

# Instanciate a Gaussian Process model
kernel = C(1.0, (1e-3, 1e3)) * RBF(10, (1e-2, 1e2))
gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=9)

# Fit to data using Maximum Likelihood Estimation of the parameters
gp.fit(X_train, y_train)

# Importing the data testing
data_test = pd.read_csv('NEW-DATA-2.T15.csv')

data_test = data_test.drop(columns=['1:Date'])
data_test = data_test.drop(columns=['2:Time'])
data_test = data_test.drop(columns=['19:Exterior_Entalpic_1'])
data_test = data_test.drop(columns=['20:Exterior_Entalpic_2'])
data_test = data_test.drop(columns=['21:Exterior_Entalpic_turbo'])
data_test = data_test.drop(columns=['24:Day_Of_Week'])

# Number of columns
print(len(data_test.columns))
print(data_test.columns)

X_test = data_test.iloc[:, [0]].values
y_test = data_test.iloc[:, 1].values

# Taking care of missing data
from sklearn.preprocessing import Imputer
# Kita dpt menggunakan Object Inspector utk melihat parameternya
imputer = Imputer(missing_values= 'NaN', strategy = 'mean', axis = 0)
# Axis mana yang ingin kita perbaiki dengan Object Imputer
imputer = imputer.fit(X_test[:, :])
# Selanjutnya kita pilih axis mana yg ingin kita re-place dgn data yg baru
X_test[:, :] = imputer.transform(X_test[:, :])

# Make the prediction on the meshed x-axis (ask for MSE as well)
y_pred, sigma = gp.predict(X_test, return_std=True)

from sklearn.metrics import explained_variance_score, mean_absolute_error, mean_squared_error, median_absolute_error, r2_score
# The best possible score is 1.0, lower values are worse.
print("Score of variance : {}".format(explained_variance_score(y_test, y_pred)))
print("Score of mean absolute error : {}".format( mean_absolute_error(y_test, y_pred)))
print("Score of mean squared error : {}".format(mean_squared_error(y_test, y_pred)))
print("Score of median absolute error : {}".format(median_absolute_error(y_test, y_pred)))
print("Score of r2 score : {}".format(r2_score(y_test, y_pred)))

# Save the result into raw files
y_pred = y_pred.tolist()
y_test = y_test.tolist()

for element in y_pred:
    with open("predict_temp", "w", encoding="utf8") as file_open:
        file_open.write(str(element))
        file_open.write('\n')
        file_open.close()

for element in y_test:
    with open("real_temp", "w", encoding="utf8") as file_open:
        file_open.write(str(element))
        file_open.write('\n')
        file_open.close()

# Visualizing the Training set results
plt.plot(y_test, color = 'blue')
plt.plot(y_pred, color = 'red')
plt.title('Prediction VS Real')
plt.xlabel('Time')
plt.ylabel('Temperature')
plt.show()
