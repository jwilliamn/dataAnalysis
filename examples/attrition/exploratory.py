#!/usr/bin/env python
# coding: utf-8

"""
    Bbva Data Challenge
    ============================
    Machine learning model to predict the customer attrition probability
    
    > Exploration of variables

    Structure:
        exploratory.py
        attrition_app_utils.py
        attritionBbvaNN.py
        input/
            train_clientes.csv    
            train_requerimientos.csv
            test_clientes.csv
            test_requerimientos.csv

    _copyright_ = 'Copyright (c) 2017 J.W.', see AUTHORS for more details
    _license_ = GNU General Public License, see LICENSE for more details
"""

import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from fancyimpute import MICE

import matplotlib.pyplot as plt



# Reading data ####
train = pd.read_csv('input/train_clientes.csv')
test = pd.read_csv('input/test_clientes.csv')

train_req = pd.read_csv('input/train_requerimientos.csv')
test_req = pd.read_csv('input/test_requerimientos.csv')


# Exploratory Analysis ####
trainTmp = train.copy()
trainTmp = trainTmp.drop(['ATTRITION'], axis=1)


data = pd.concat([trainTmp, test])

data.describe()
data.isnull().any()
data.std()
data.dtypes

# Checking which variables are categories 'data.dtypes': RANG_INGRESO, FLAG_LIMA_PROVINCIA, 
# RANG_SDO_PASIVO_MENOS0, RANG_NRO_PRODUCTOS_MENOS0 then map to numbers.
data['RANG_INGRESO'] = data['RANG_INGRESO'].map({'Rang_ingreso_01':1, 'Rang_ingreso_02':2, 'Rang_ingreso_03':3, 
                                                 'Rang_ingreso_04':4, 'Rang_ingreso_05':5, 'Rang_ingreso_06':6,
                                                 'Rang_ingreso_07':7, 'Rang_ingreso_08':8, 'Rang_ingreso_09':9})
data['FLAG_LIMA_PROVINCIA'] = data['FLAG_LIMA_PROVINCIA'].map({'Lima':1, 'Provincia':0})
data['RANG_SDO_PASIVO_MENOS0'] = data['RANG_SDO_PASIVO_MENOS0'].map({'Cero':0, 'Rango_SDO_01':1, 'Rango_SDO_02':2,
                                    'Rango_SDO_04':4, 'Rango_SDO_03':3, 'Rango_SDO_05':5, 'Rango_SDO_06':6, 'Rango_SDO_07':7, 
                                    'Rango_SDO_08':8, 'Rango_SDO_14':14, 'Rango_SDO_13':13, 'Rango_SDO_09':9, 'Rango_SDO_11':11, 
                                    'Rango_SDO_10':10, 'Rango_SDO_12':12})
data['RANG_NRO_PRODUCTOS_MENOS0'] = data['RANG_NRO_PRODUCTOS_MENOS0'].map({'Rango_02':2, 'Rango_03':3, 'Rango_04':4, 
                                                 'Rango_01':1, 'Rango_05':5, 'Rango_06':6})
#data.loc[data["RANG_INGRESO"] == 'Rang_ingreso_01', 'RANG_INGRESO'] = 1
#data["RANG_INGRESO"].astype('category').cat.categories
#data['RANG_INGRESO'].value_counts()


# Cleaning data ####
# Dropping CODMES bc std = 0 (same value for all examples) 'data.std()'
data = data.drop(["CODMES"], axis=1)


# Data imputation ####
# train = train.dropna(axis=0)  # dataset without missing values
# test = test.dropna(axis=0)

# Numerical variable, replacing 'nan' values with the mean
#train["EDAD"] = train["EDAD"].fillna(int(train["EDAD"].mean()))  # int(train["EDAD"].mean()) <<<
#train["ANTIGUEDAD"] = train["ANTIGUEDAD"].fillna(int(train["ANTIGUEDAD"].mean()))
#test["EDAD"] = test["EDAD"].fillna(int(test["EDAD"].mean()))
#test["ANTIGUEDAD"] = test["ANTIGUEDAD"].fillna(int(test["ANTIGUEDAD"].mean()))

# Categorical variable, replacing missing values with the most frequent value
#train = train.apply(lambda x:x.fillna(x.value_counts().index[0]))  
#test = test.apply(lambda x:x.fillna(x.value_counts().index[0]))  

x = data.as_matrix()
x_filled = MICE().complete(x)

# Imputed values to int
#x_tmp = x_filled[:,3].astype(int)  # just one column
x_filled = x_filled.astype(int)
x_filled = pd.DataFrame(x_filled)
x_filled.columns = data.columns

# Merge with second table - Adding new features
data_req = pd.concat([train_req, test_req])
data_req_tmp = data_req[['ID_CORRELATIVO', 'DICTAMEN']].groupby(['ID_CORRELATIVO'], as_index=False).agg(lambda x:x.value_counts().index[0])
#len(np.unique(data_req_tmp['ID_CORRELATIVO']))
#List unique values in the df['name'] column
#df.name.unique()
data = pd.merge(x_filled, data_req_tmp, how='left', on='ID_CORRELATIVO')
#data_ = pd.get_dummies(data)

# Encoding categorical variables ####
# RANG_INGRESO, RANG_SDO_PASIVO_MENOS0, RANG_NRO_PRODUCTOS_MENOS0, converting to onehot variables
data['RANG_INGRESO'] = data['RANG_INGRESO'].astype(object)

data['RANG_SDO_PASIVO_MENOS0'] = data['RANG_SDO_PASIVO_MENOS0'].astype(object)
data['RANG_NRO_PRODUCTOS_MENOS0'] = data['RANG_NRO_PRODUCTOS_MENOS0'].astype(object)
data = pd.get_dummies(data)

data = data.astype(int)

# Split data to the original train and test set
train_ = data[0:70000].copy()
test_ = data[70000:100000].copy()

Xtest = test_.drop(['ID_CORRELATIVO'], axis=1)
Xtest = Xtest.as_matrix()

X = train_.drop(['ID_CORRELATIVO'], axis=1)
X = X.as_matrix()

Y = train['ATTRITION']
Y = np.array(Y)

# Assuming train_ as the whole dataset, then split it into tiny train and test set
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.15)

# Feature scaling ####
scaler = StandardScaler().fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

Xtest = scaler.transform(Xtest)


# Reshaping data to feed the network
X_train = X_train.T
X_test = X_test.T
Xtest = Xtest.T

y_train = y_train.reshape((1, len(y_train)))
y_test = y_test.reshape((1, len(y_test)))


































