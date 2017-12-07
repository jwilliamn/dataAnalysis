#!/usr/bin/env python
# coding: utf-8

"""
    Bbva Data Challenge
    ============================
    Machine learning model to predict the customer attrition probability 

    Structure:
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
from attrition_app_utils import *



# Input ##
train = pd.read_csv('input/train_clientes.csv')
test = pd.read_csv('input/test_clientes.csv')


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

# Separating training data into features and labels
X = train.drop(['ID_CORRELATIVO', 'ATTRITION'], axis=1)
Y = train['ATTRITION']

X = X.as_matrix()
Y = np.array(Y)

# Formatting test data
X_test = test.drop(['ID_CORRELATIVO'], axis=1)
X_test = X_test.as_matrix()

# Splitting the data (if applicable) into training and test set (subset)
#X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2)
X_train, y_train = X, Y

# Feature scaling
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.fit_transform(X_test)


# Appropiate Reshaping to feed the network
X_train = X_train.T
X_test = X_test.T

y_train = y_train.reshape((1, len(y_train)))
#y_test = y_test.reshape((1, len(y_test)))

# Modeling ##
# Hyperparameters setting - model design
dropout = 0.1
epochs = 100
batch_size = 30
optimizer = 'adam'
k = 20

### CONSTANTS DEFINING THE MODEL ####
layers_dims = [50, 64, 32, 4, 2, 1]  #[50, 64, 32, 4, 2, 1]  # 6-layer model & lr=0.08
lr = 0.08  #lr was 0.009,

# L-layer Neural Network
def L_layer_model(X, Y, layers_dims, learning_rate = 0.05, num_iterations = 3000, print_cost=False):  #lr was 0.009
    """
    L-layer neural network: 
    Arquitecture: [LINEAR->RELU]*(L-1)->LINEAR->SIGMOID.
    
    Arguments:
    X -- data, numpy array of shape (features, number of examples)
    Y -- true "label" vector (containing 0 if non-Attrition, 1 if Attrition), of shape (1, number of examples)
    layers_dims -- list containing the input size and each layer size, of length (number of layers + 1).
    learning_rate -- learning rate of the gradient descent update rule
    num_iterations -- number of iterations of the optimization loop
    print_cost -- if True, it prints the cost every 100 steps
    
    Returns:
    parameters -- parameters learnt by the model. They can then be used to predict.
    """

    np.random.seed(1)
    costs = []                         # keep track of cost
    
    # Parameters initialization.
    parameters = initialize_parameters_deep(layers_dims)
    
    # Loop (gradient descent)
    for i in range(0, num_iterations):
        # Forward propagation: [LINEAR -> RELU]*(L-1) -> LINEAR -> SIGMOID.
        AL, caches = L_model_forward(X, parameters)
        
        # Compute cost.
        cost = compute_cost(AL, Y)
    
        # Backward propagation.
        grads = L_model_backward(AL, Y, caches)
 
        # Update parameters.
        parameters = update_parameters(parameters, grads, learning_rate)
                
        # Print the cost every 500 training example
        if print_cost and i % 500 == 0:
            print ("Cost after iteration %i: %f" %(i, cost))
        if print_cost and i % 100 == 0:
            costs.append(cost)
            
    # plot the cost
    plt.plot(np.squeeze(costs))
    plt.ylabel('cost')
    plt.xlabel('iterations (per tens)')
    plt.title("Learning rate =" + str(learning_rate))
    plt.show()
    
    return parameters


# Training the model as a 6-layer neural network. 
parameters = L_layer_model(X_train, y_train, layers_dims, learning_rate=lr, num_iterations = 30000, print_cost = True)

pred_train, prob_train = predict(X_train, y_train, parameters)
#pred_test, prob_test = predict(X_test, y_test, parameters)
prob_test, _ = L_model_forward(X_test, parameters)

# Saving probabilities to send for testing
#train['prob_'] = prob_train.T
test['ATTRITION'] = prob_test.T

send = test[['ID_CORRELATIVO','ATTRITION']]

send.to_csv("output/testProb.csv", index=False)









































