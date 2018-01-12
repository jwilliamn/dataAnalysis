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
import math
import matplotlib.pyplot as plt

import xgboost as xgb
import tensorflow as tf

from sklearn.model_selection import GridSearchCV, StratifiedKFold
from exploratory import X_train, X_test, y_train, y_test, Xtest, train_


def compute_cost(AL, Y):
    """
    Arguments:
    AL -- probability vector corresponding to your label predictions, shape (1, number of examples)
    Y -- true "label" vector (for example: containing 0 if non-cat, 1 if cat), shape (1, number of examples)

    Returns:
    cost -- cross-entropy cost
    """
    
    m = Y.shape[1]

    # Compute loss from aL and y.
    cost = (1./m) * (-np.dot(Y,np.log(AL).T) - np.dot(1-Y, np.log(1-AL).T))
    
    cost = np.squeeze(cost)
    assert(cost.shape == ())
    
    return cost

def random_mini_batches(X, mini_batch_size = 64, seed = 0):
    """
    Creates a list of random minibatches from (X, Y)
    
    Arguments:
    X -- input data, of shape (input size, number of examples)
    mini_batch_size - size of the mini-batches, integer
    seed -- this is only for the purpose of grading, so that you're "random minibatches are the same as ours.
    
    Returns:
    mini_batches -- list of synchronous (mini_batch_X)
    """
    
    X = X.T
    
    m = X.shape[1]                  # number of training examples
    mini_batches = []
    np.random.seed(seed)
    
    # Step 1: Shuffle (X, Y)
    permutation = list(np.random.permutation(m))
    shuffled_X = X[:, permutation]
    
    # Step 2: Partition (shuffled_X, shuffled_Y). Minus the end case.
    num_complete_minibatches = math.floor(m/mini_batch_size) # number of mini batches of size mini_batch_size in your partitionning
    for k in range(0, num_complete_minibatches):
        mini_batch_X = shuffled_X[:, k * mini_batch_size : k * mini_batch_size + mini_batch_size]
        mini_batch = (mini_batch_X.T)
        mini_batches.append(mini_batch)
    
    # Handling the end case (last mini-batch < mini_batch_size)
    if m % mini_batch_size != 0:
        mini_batch_X = shuffled_X[:, num_complete_minibatches * mini_batch_size : m]
        mini_batch = (mini_batch_X.T)
        mini_batches.append(mini_batch)
    
    return mini_batches


X_train, X_test, Xtest = X_train.T, X_test.T, Xtest.T
y_train, y_test = np.ravel(y_train.T), np.ravel(y_test.T)
#plt.matshow(pd.DataFrame(X_train).corr())

# Autoencoder 
# Training parameters
learning_ratenn = 0.01
num_epochs = 3000
minibatch_size = 64

# Network parameters
nh_1 = 64
nh_2 = 32
#num_i = 80

(m, num_i) = X_train.shape

# Placeholder
X = tf.placeholder(dtype=tf.float32, shape=[None, num_i], name = 'X_placeholder')

# Initialize parameters
weights = {
        'encoder_h1': tf.get_variable(name="W1", shape=[num_i, nh_1], initializer = tf.contrib.layers.xavier_initializer(seed = 1)),
        'encoder_h2': tf.get_variable(name="W2", shape=[nh_1, nh_2], initializer = tf.contrib.layers.xavier_initializer(seed = 1)),
        'decoder_h1': tf.get_variable(name="W1p", shape=[nh_2, nh_1], initializer = tf.contrib.layers.xavier_initializer(seed = 1)),
        'decoder_h2': tf.get_variable(name="W2p", shape=[nh_1, num_i], initializer = tf.contrib.layers.xavier_initializer(seed = 1))
        }

biases = {
        'encoder_b1': tf.get_variable(name="b1", shape=[1, nh_1], initializer = tf.contrib.layers.xavier_initializer(seed = 1)),
        'encoder_b2': tf.get_variable(name="b2", shape=[1, nh_2], initializer = tf.contrib.layers.xavier_initializer(seed = 1)),
        'decoder_b1': tf.get_variable(name="b1p", shape=[1, nh_1], initializer = tf.contrib.layers.xavier_initializer(seed = 1)),
        'decoder_b2': tf.get_variable(name="b2p", shape=[1, num_i], initializer = tf.contrib.layers.xavier_initializer(seed = 1))
        }

# Building the encoder
def encoder(x):
    layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['encoder_h1']),
                                   biases['encoder_b1']))
    layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, weights['encoder_h2']),
                                   biases['encoder_b2']))
    return layer_2

# Building the decoder
def decoder(x):
    layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['decoder_h1']),
                                   biases['decoder_b1']))
    layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, weights['decoder_h2']),
                                   biases['decoder_b2']))
    return layer_2

# Constructing the model
encoder_op = encoder(X)
decoder_op = decoder(encoder_op)

# Prediction
y_pred = decoder_op
y_true = X

# Define the loss and optimizer
loss = tf.reduce_mean(tf.pow(y_true - y_pred, 2))
optimizer = tf.train.RMSPropOptimizer(learning_ratenn).minimize(loss)

# Initialize variables
init = tf.global_variables_initializer()
costs = [] 

# Start training
with tf.Session() as sess:
    # Run the initilizer
    sess.run(init)
    
    # Do the training loop
    for epoch in range(num_epochs):

        epoch_cost = 0.                       # Defines a cost related to an epoch
        num_minibatches = int(m / minibatch_size) # number of minibatches of size minibatch_size in the train set
        minibatches = random_mini_batches(X_train, minibatch_size)

        for minibatch in minibatches:

            # Select a minibatch
            (minibatch_X) = minibatch
            
            # IMPORTANT: The line that runs the graph on a minibatch.
            # Run the session to execute the "optimizer" and the "cost", the feedict should contain a minibatch for (X,Y).
            
            _ , minibatch_cost = sess.run([optimizer, loss], feed_dict={X:minibatch_X})
            
            
            epoch_cost += minibatch_cost / num_minibatches

        # Print the cost every epoch
        if epoch % 100 == 0:
            print ("Cost after epoch %i: %f" % (epoch, epoch_cost))
        if epoch % 5 == 0:
            costs.append(epoch_cost)
    
    # plot the cost
    plt.plot(np.squeeze(costs))
    plt.ylabel('cost')
    plt.xlabel('iterations (per tens)')
    plt.title("Learning rate =" + str(learning_ratenn))
    plt.show()
    
    # lets save the parameters in a variable
    X_denoised = sess.run(decoder_op, feed_dict={X:X_train})
    print ("X_denoised have been trained!")




# Instantiate the XGBClassifier: xg_cl
xg_cl = xgb.XGBClassifier(learning_rate=0.09, max_depth=6, objective='binary:logistic', n_estimators=1200, seed=123)
#xg_cl = xgb.XGBClassifier()


## Grid search - hyperparameter tunning
#learning_rate = [0.0001, 0.001, 0.01, 0.1, 0.2, 0.3]
#param_grid = dict(learning_rate=learning_rate)
#
#n_estimators = [50, 100, 150, 200, 1000]
#max_depth = [2, 4, 6, 8]
#param_grid = dict(max_depth=max_depth, n_estimators=n_estimators)
#
#kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=7)
#grid_search = GridSearchCV(xg_cl, param_grid, scoring="neg_log_loss", n_jobs=-1, cv=kfold, verbose=1)
#grid_result = grid_search.fit(X_train, y_train)
## summarize results
#print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
#means = grid_result.cv_results_['mean_test_score']
#stds = grid_result.cv_results_['std_test_score']
#params = grid_result.cv_results_['params']
#for mean, stdev, param in zip(means, stds, params):
#	print("%f (%f) with: %r" % (mean, stdev, param))


# Fit the classifier to the training set
eval_set = [(X_test, y_test)]
xg_cl.fit(X_denoised, y_train, early_stopping_rounds=40, eval_metric="logloss", eval_set=eval_set, verbose=True)

print(xg_cl.feature_importances_)
xgb.plot_importance(xg_cl)
plt.show()

xgb.plot_tree(xg_cl)
plt.show()


# cost of training
preds_train = xg_cl.predict(X_train)
probs_train = xg_cl.predict_proba(X_train)
probs_tr = probs_train[:,1]
probs_tr = probs_tr.reshape((1, len(probs_tr)))
y_train_ = y_train.reshape((1, len(y_train)))

cost_train = compute_cost(probs_tr, y_train_)
print("Training cost: %f" % cost_train)

# Predict the labels of the test set: preds
preds = xg_cl.predict(X_test)
probs = xg_cl.predict_proba(X_test)
#probs_ = np.max(probs, axis=1)
probs_ = probs[:,1]
probs_ = probs_.reshape((1, len(probs_)))
y_test_ = y_test.reshape((1, len(y_test)))

cost_test = compute_cost(probs_, y_test_)
print("Test cost: %f" % cost_test)

# Real test set probs
probs_send = xg_cl.predict_proba(Xtest)
probs_se = probs_send[:,1]

# Compute the accuracy: accuracy
accuracy_train = float(np.sum(preds_train==y_train))/y_train.shape[0]
print("accuracy train: %f" % (accuracy_train))

accuracy = float(np.sum(preds==y_test))/y_test.shape[0]
print("accuracy: %f" % (accuracy))

# Send file
test['ATTRITION'] = probs_se

send = test[['ID_CORRELATIVO','ATTRITION']]

send.to_csv("output/testProbx.csv", index=False)


#Training cost: 0.298948
#Test cost: 0.315317

#Training cost: 0.282102
#Test cost: 0.310029
#accuracy train: 0.880036
#accuracy: 0.873357

## Create the DMatrix: churn_dmatrix
#churn_dmatrix = xgb.DMatrix(data=X, label=y)
#
## Create the parameter dictionary: params
#params = {"objective":"reg:logistic", "max_depth":3}
#
## Perform cross-validation: cv_results
#cv_results = xgb.cv(dtrain=churn_dmatrix, params=params, nfold=3, num_boost_round=5, metrics="error", as_pandas=True, seed=123)
#
## Print cv_results
#print(cv_results)
#
## Print the accuracy
#print(((1-cv_results["test-error-mean"]).iloc[-1]))