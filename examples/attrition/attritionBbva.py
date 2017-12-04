#!/usr/bin/env python
# coding: utf-8

"""
    Bbva Data Challenge
    ============================
    Machine learning model to predict the customer attrition probability 

    Structure:
        attritionBbva.py
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

from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from keras.models import Sequential
from keras.layers import Dense, Dropout


# Input ##
train = pd.read_csv('input/train_clientes.csv')


# Exploratory Analysis ##
train.describe()
train.isnull().any()
train.std()

# Cleaning data
# Dropping CODMES bc std = 0 (same value for all examples)
train = train.drop(["CODMES"], axis=1)

# Dealing with NsN values
train["RANG_INGRESO"].isnull().sum()
train["FLAG_LIMA_PROVINCIA"].isnull().sum()
train["EDAD"].isnull().sum()
train["ANTIGUEDAD"].isnull().sum()

train = train.dropna(axis=0)  # dataset without missing values

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

# Separating data into features and labels
X = train.drop(['ID_CORRELATIVO', 'ATTRITION'], axis=1)
Y = train['ATTRITION']

X = X.as_matrix()
Y = np.array(Y)

# Splitting the data into training and test set (subset)
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2)

# Feature scaling
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


# Modeling ##
# Hyperparameters setting - model design
dropout = 0.1
epochs = 100
batch_size = 30
optimizer = 'adam'
k = 20

# Model building
def build_classifier():
    classifier = Sequential()
    classifier.add(Dense(16, kernel_initializer="truncated_normal", activation = 'relu', input_shape = (X.shape[1],)))
    classifier.add(Dropout(dropout))
    classifier.add(Dense(1, kernel_initializer="truncated_normal", activation = 'sigmoid', )) #outputlayer
    classifier.compile(optimizer = optimizer, loss = 'binary_crossentropy', metrics = ["accuracy"])
    return classifier

# Model training
classifier = KerasClassifier(build_fn = build_classifier, batch_size = batch_size, epochs = epochs, verbose=1)
accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 30)
max = accuracies.max()
print("Best accuracy: ",max)




