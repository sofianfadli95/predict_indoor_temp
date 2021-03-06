# -*- coding: utf-8 -*-
"""
Created on Sun Jun 24 10:09:33 2018

@author: sofyan.fadli
"""

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

# Feature Scaling 
from sklearn.preprocessing import StandardScaler 

sc_X = StandardScaler() 
# sc_Y = StandardScaler().fit(y_train) 
X_train = sc_X.fit_transform(X_train) 
# y_train = sc_Y.transform(y_train) 

# Building the optimal model using Backward Elimination 
import statsmodels.formula.api as sm 

def backwardElimination(x, sl): 
    numVars = len(x[0]) 
    for i in range(0, numVars): 
        regressor_OLS = sm.OLS(y_train, X_train).fit() 
        maxVar = max(regressor_OLS.pvalues).astype(float) 
        if maxVar > sl: 
            for j in range(0, numVars - i): 
                if (regressor_OLS.pvalues[j].astype(float) == maxVar): 
                    x = np.delete(x, j, 1) 
    regressor_OLS.summary() 
    return x 
  
SL = 0.05 
X_opt = X_train[:, [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16]] 
X_Modeled = backwardElimination(X_opt, SL) 

# Applying PCA 
# Dengan menggunakan PCA, skrg independent variable kita hanya menjadi 2 saja 
from sklearn.decomposition import PCA 
pca = PCA(n_components = 2) 
X_Modeled = pca.fit_transform(X_Modeled) 
# X_test = pca.transform(X_test) 
explained_variance = pca.explained_variance_ratio_ 

##################################### Testing ##################################### 
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

X_test = data_test.iloc[:, [0,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17]].values 
y_test = data_test.iloc[:, 1].values 

# Taking care of missing data 
from sklearn.preprocessing import Imputer 
# Kita dpt menggunakan Object Inspector utk melihat parameternya 
imputer = Imputer(missing_values= 'NaN', strategy = 'mean', axis = 0) 
# Axis mana yang ingin kita perbaiki dengan Object Imputer 
imputer = imputer.fit(X_test[:, :]) 
# Selanjutnya kita pilih axis mana yg ingin kita re-place dgn data yg baru 
X_test[:, :] = imputer.transform(X_test[:, :]) 

sc_X = StandardScaler() 
# sc_Y = StandardScaler().fit(y_test) 
X_test = sc_X.fit_transform(X_test) 
# y_test = sc_Y.transform(y_test) 
  
SL = 0.05 
X_opt_test = X_test[:, [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16]] 
X_Modeled_test = backwardElimination(X_opt_test, SL) 

# Applying PCA 
from sklearn.decomposition import PCA 
pca = PCA(n_components = 2) 
X_Modeled_test = pca.fit_transform(X_Modeled_test) 

# Prediction using Gaussian Process 
# data for merging 
# data_merge = pd.read_csv('result_lstm_min_100.csv') 
# data_merge = data_merge[['date', 'time','real_temp']] 

########################################## George Kernel ################################### 
import george 
from george.kernels import Matern32Kernel, ConstantKernel, ExpSquaredKernel

# kernel = ConstantKernel(0.5, ndim=2) * Matern32Kernel(0.5, ndim=2)
kernel = 10.0 * ExpSquaredKernel([0.5, 0.5], ndim=2, axes=[0,1]) # Lumayan nih...
print("Kernel parameters names : {}".format(kernel.get_parameter_names()))
print("Kernel parameters names : {}".format(kernel.get_parameter_vector())) 
gp = george.GP(kernel) 
gp.compute(X_Modeled) 
y_pred, cov = gp.predict(y_train, X_Modeled_test) 

from sklearn.metrics import explained_variance_score, mean_absolute_error, mean_squared_error, median_absolute_error, r2_score 
# The best possible score is 1.0, lower values are worse. 
print("Score of variance : {}".format(explained_variance_score(y_test, y_pred))) 
print("Score of mean absolute error : {}".format( mean_absolute_error(y_test, y_pred))) 
print("Score of mean squared error : {}".format(mean_squared_error(y_test, y_pred))) 
print("Score of median absolute error : {}".format(median_absolute_error(y_test, y_pred))) 
print("Score of r2 score : {}".format(r2_score(y_test, y_pred))) 

result_gp_george = pd.DataFrame(y_pred).rename(columns = {0:'predicted_temp'}) 
# result_gp_george = pd.concat((data_merge, result_gp_george), axis = 1) 
result_gp_george.to_csv('res_gp_george_time.csv') 

result_gp_george = result_gp_george[['date','real_temp','predicted_temp']].groupby('date', as_index= False).mean().reset_index(drop=True) 
result_gp_george.to_csv('res_gp_george_day.csv')

# Visualizing the Training set results
plt.plot(y_test, color = 'blue')
plt.plot(y_pred, color = 'red')
plt.title('Prediction VS Real')
plt.xlabel('Time')
plt.ylabel('Temperature')
plt.show()

######################################## Done George Kernel #################################