#!/usr/bin/env python
# coding: utf-8

"""
    Bbva Data Challenge
    ============================
    Machine learning model to predict the customer attrition probability 

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

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt



# Reading data ##
train = pd.read_csv('input/train_clientes.csv')
test = pd.read_csv('input/test_clientes.csv')

train_req = pd.read_csv('input/train_requerimientos.csv')
test_req = pd.read_csv('input/test_requerimientos.csv')


# Exploratory Analysis ##
#train.describe()
#train.isnull().any()
#train.std()

# Cleaning data
# Dropping CODMES bc std = 0 (same value for all examples)
train = train.drop(["CODMES"], axis=1)
test= test.drop(["CODMES"], axis=1)

# Dealing with NaN values
train["RANG_INGRESO"].isnull().sum()
train["FLAG_LIMA_PROVINCIA"].isnull().sum()
train["EDAD"].isnull().sum()
train["ANTIGUEDAD"].isnull().sum()

# Data imputation
# train = train.dropna(axis=0)  # dataset without missing values
# test = test.dropna(axis=0)

# Numerical variable, replacing 'nan' values with the mean
train["EDAD"] = train["EDAD"].fillna(int(train["EDAD"].mean()))  # int(train["EDAD"].mean()) <<<
train["ANTIGUEDAD"] = train["ANTIGUEDAD"].fillna(int(train["ANTIGUEDAD"].mean()))
test["EDAD"] = test["EDAD"].fillna(int(test["EDAD"].mean()))
test["ANTIGUEDAD"] = test["ANTIGUEDAD"].fillna(int(test["ANTIGUEDAD"].mean()))

# Categorical variable, replacing missing values with the most frequent value
train = train.apply(lambda x:x.fillna(x.value_counts().index[0]))  
test = test.apply(lambda x:x.fillna(x.value_counts().index[0]))  

# Encoding categorical data 
# RANG_NRO_PRODUCTOS_MENOS0, RANG_SDO_PASIVO_MENOS0, FLAG_LIMA_PROVINCIA, 
# RANG_INGRESO, converting from string to numerical values
labelencoder_X = LabelEncoder()

#labelencoder_X.fit(train["RANG_INGRESO"])
#labelencoder_X.classes_  # 9 classes
#labelencoder_X.fit(train["FLAG_LIMA_PROVINCIA"])
#labelencoder_X.classes_  # 2 classes
#labelencoder_X.fit(train["RANG_SDO_PASIVO_MENOS0"])
#labelencoder_X.classes_  # 15 classes
#labelencoder_X.fit(train["RANG_NRO_PRODUCTOS_MENOS0"])
#labelencoder_X.classes_  # 6 classes

train["RANG_INGRESO"] = labelencoder_X.fit_transform(train["RANG_INGRESO"])
train["FLAG_LIMA_PROVINCIA"] = labelencoder_X.fit_transform(train["FLAG_LIMA_PROVINCIA"])
train["RANG_SDO_PASIVO_MENOS0"] = labelencoder_X.fit_transform(train["RANG_SDO_PASIVO_MENOS0"])
train["RANG_NRO_PRODUCTOS_MENOS0"] = labelencoder_X.fit_transform(train["RANG_NRO_PRODUCTOS_MENOS0"])

labelencoder_X_test = LabelEncoder()
test["RANG_INGRESO"] = labelencoder_X_test.fit_transform(test["RANG_INGRESO"])
test["FLAG_LIMA_PROVINCIA"] = labelencoder_X_test.fit_transform(test["FLAG_LIMA_PROVINCIA"])
test["RANG_SDO_PASIVO_MENOS0"] = labelencoder_X_test.fit_transform(test["RANG_SDO_PASIVO_MENOS0"])
test["RANG_NRO_PRODUCTOS_MENOS0"] = labelencoder_X_test.fit_transform(test["RANG_NRO_PRODUCTOS_MENOS0"])
