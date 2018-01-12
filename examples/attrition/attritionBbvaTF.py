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
import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow.python.framework import ops
from tf_utils import load_dataset, random_mini_batches, convert_to_one_hot #, predict
from exploratory import X_train, X_test, y_train, y_test, Xtest


y_test_alt = y_test.copy()
# GLobal settings ####
# Network Hyperparameters
# Number of cells in hidden layer 1,...
#N1 = 80  
#N2 = 32
#N3 = 16  # might delete this
#N4 = 8
#N5 = 4
#N6 = 2

#N1 = 32
#N2 = 16
#N3 = 64
#N4 = 32
#N5 = 16
#N6 = 4

N1 = 32
N2 = 16
N3 = 8


# Functions of the model
def create_placeholders(n_x, n_y):
    X = tf.placeholder(dtype=tf.float32, shape=[n_x, None], name = 'X_placeholder')
    Y = tf.placeholder(dtype=tf.float32, shape=[n_y, None], name = 'Y_placeholder')
    return X, Y

X_teste = tf.constant(X_test, dtype= tf.float32)
Xteste = tf.constant(Xtest, dtype=tf.float32)
y_test_alt = tf.constant(y_test_alt, dtype=tf.float32)


def initialize_parameters(n_x):
    """
    Initializes parameters to build a neural network with tensorflow.
    
    Returns:
    parameters -- a dictionary of tensors containing W1, b1, W2, b2, W3, b3, ...
    """
    
    W1 = tf.get_variable(name="W1", shape=[N1, n_x], initializer = tf.contrib.layers.xavier_initializer(seed = 1))
    b1 = tf.get_variable(name="b1", shape=[N1, 1], initializer = tf.zeros_initializer())
    W2 = tf.get_variable(name="W2", shape=[N2, N1], initializer = tf.contrib.layers.xavier_initializer(seed = 1))
    b2 = tf.get_variable(name="b2", shape=[N2, 1], initializer = tf.zeros_initializer())
    W3 = tf.get_variable(name="W3", shape=[N3, N2], initializer = tf.contrib.layers.xavier_initializer(seed = 1))
    b3 = tf.get_variable(name="b3", shape=[N3, 1], initializer = tf.zeros_initializer())
    W7 = tf.get_variable(name="W7", shape=[1, N3], initializer = tf.contrib.layers.xavier_initializer(seed = 1))
    b7 = tf.get_variable(name="b7", shape=[1, 1], initializer = tf.zeros_initializer())
    
    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2,
                  "W3": W3,
                  "b3": b3,
                  "W7": W7,
                  "b7": b7}
    
    return parameters

def forward_propagation(X, parameters):
    """
    Implements the forward propagation for the model: LINEAR -> RELU -> LINEAR -> RELU -> LINEAR -> SOFTMAX
    
    Arguments:
    X -- input dataset placeholder, of shape (input size, number of examples)
    parameters -- python dictionary containing your parameters "W1", "b1", "W2", "b2", "W3", "b3"
                  the shapes are given in initialize_parameters

    Returns:
    Z5 -- the output of the last LINEAR unit
    """
    
    # Retrieve the parameters from the dictionary "parameters" 
    W1 = parameters['W1']
    b1 = parameters['b1']
    W2 = parameters['W2']
    b2 = parameters['b2']
    W3 = parameters['W3']
    b3 = parameters['b3']
    W7 = parameters['W7']
    b7 = parameters['b7']
    
    # Forward propagation
    Z1 = tf.add(tf.matmul(W1, X), b1)
    A1 = tf.nn.relu(Z1)
    Z2 = tf.add(tf.matmul(W2, A1), b2)
    A2 = tf.nn.relu(Z2)
    Z3 = tf.add(tf.matmul(W3, A2), b3)
    A3 = tf.nn.relu(Z3)
    A3 = tf.nn.dropout(A3, keep_prob=0.5)
    oL = tf.add(tf.matmul(W7, A3), b7)
    
    return oL

def compute_cost(oL, Y):
    """
    Computes the cost
    
    Arguments:
    Z3 -- output of forward propagation (output of the last LINEAR unit), of shape (6, number of examples)
    Y -- "true" labels vector placeholder, same shape as Z3
    
    Returns:
    cost - Tensor of the cost function
    """
    
    logits = tf.transpose(oL)
    labels = tf.transpose(Y)
    
    
    cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=labels))
    
    return cost

def predict(probs):    
    m = probs.shape[0]
    print("m: ", m)
    print(" probs.shape ", probs.shape)
    p = np.zeros((m,1))
    
    # convert probas to 0/1 predictions
    for i in range(0, probs.shape[0]):
        if probs[i,0] > 0.5:
            p[i,0] = 1
        else:
            p[i,0] = 0
    
    #print results
    #print ("predictions: " + str(p))
    #print ("true labels: " + str(y))
    #print("Accuracy: "  + str(np.sum((p == y)/m)))
        
    return p, probs


def accuracy2(predictions, labels):
    return np.sum((predictions == labels)/predictions.shape[0])

#Y_train, Y_test = y_train, y_test
#
#learning_rate = 0.01
#num_epochs = 1
#minibatch_size = 32
#print_cost = True

# Model design ####
def model(X_train, Y_train, X_teste, Y_test, Y_test_alt, Xteste, learning_rate = 0.001,
          num_epochs = 1200, minibatch_size = 256, print_cost = True):
    """
    Implements a five-layer tensorflow neural network: 
        LINEAR->RELU->LINEAR->RELU->LINEAR->SOFTMAX.
    
    Arguments:
    X_train -- training set, of shape (input size = , number of training examples = )
    Y_train -- test set, of shape (output size = , number of training examples = )
    X_test -- training set, of shape (input size = , number of training examples = )
    Y_test -- test set, of shape (output size = , number of test examples = )
    learning_rate -- learning rate of the optimization
    num_epochs -- number of epochs of the optimization loop
    minibatch_size -- size of a minibatch
    print_cost -- True to print the cost every 100 epochs
    
    Returns:
    parameters -- parameters learnt by the model. They can then be used to predict.
    """
    
    #ops.reset_default_graph()                         # to be able to rerun the model without overwriting tf variables
    tf.set_random_seed(1)                             # to keep consistent results
    seed = 3                                          # to keep consistent results
    (n_x, m) = X_train.shape                          # (n_x: input size, m : number of examples in the train set)
    n_y = Y_train.shape[0]                            # n_y : output size
    costs = []                                        # To keep track of the cost
    
    # Placeholders
    X, Y = create_placeholders(n_x, n_y)

    # Initialize parameters
    parameters = initialize_parameters(n_x)
    
    # Forward propagation: Build the forward propagation in the tensorflow graph
    oL = forward_propagation(X, parameters)
    
    train_prob = tf.nn.sigmoid(x= tf.transpose(oL))
    
    # Cost function: Add cost function to tensorflow graph
    cost = compute_cost(oL, Y)
    
    # Backpropagation: Using AdamOptimizer.
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
    
    cost_test = compute_cost(forward_propagation(X_teste, parameters), Y_test_alt)
    test_prob = tf.nn.sigmoid(x= tf.transpose(forward_propagation(X_teste, parameters)))
    realTest_prob = tf.nn.sigmoid(x= tf.transpose(forward_propagation(Xteste, parameters)))
    # Initialize all the variables
    init = tf.global_variables_initializer()

    # Start the session to compute the tensorflow graph
    with tf.Session() as sess:
        
        # Run the initialization
        sess.run(init)
        
        # Do the training loop
        for epoch in range(num_epochs):

            epoch_cost = 0.                       # Defines a cost related to an epoch
            num_minibatches = int(m / minibatch_size) # number of minibatches of size minibatch_size in the train set
            minibatches = random_mini_batches(X_train, Y_train, minibatch_size, seed)

            for minibatch in minibatches:

                # Select a minibatch
                (minibatch_X, minibatch_Y) = minibatch
                
                # IMPORTANT: The line that runs the graph on a minibatch.
                # Run the session to execute the "optimizer" and the "cost", the feedict should contain a minibatch for (X,Y).
                
                _ , minibatch_cost, probs = sess.run([optimizer, cost, train_prob], feed_dict={X:minibatch_X, Y:minibatch_Y})
                
                
                epoch_cost += minibatch_cost / num_minibatches

            # Print the cost every epoch
            if print_cost == True and epoch % 100 == 0:
                print ("Cost after epoch %i: %f" % (epoch, epoch_cost))
            if print_cost == True and epoch % 5 == 0:
                costs.append(epoch_cost)
                
        # plot the cost
        plt.plot(np.squeeze(costs))
        plt.ylabel('cost')
        plt.xlabel('iterations (per tens)')
        plt.title("Learning rate =" + str(learning_rate))
        plt.show()

        # lets save the parameters in a variable
        parameters = sess.run(parameters)
        print ("Parameters have been trained!")

        # Calculate the correct predictions
        pred, probs = predict(train_prob.eval({X: X_train, Y: Y_train}))
        pred_test, probs_test = predict(test_prob.eval())
        
        prob_realtest = realTest_prob.eval()
        #correct_prediction = tf.equal(pred, Y)

        # Calculate accuracy on the test set
        #accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

        #print ("Train Accuracy:", accuracy.eval({X: X_train, Y: Y_train}))
        #print ("Test Accuracy:", accuracy.eval({X: X_teste, Y: Y_test}))
        print("Train Accuracy2:", accuracy2(pred, Y_train.T))
        print("Test Accuracy2:", accuracy2(pred_test, Y_test.T)) #Y_test.T
        print("Test Cost:", cost_test.eval())
        
        return parameters, pred, probs, prob_realtest




parameters, pred, probs, prob_realtest = model(X_train, y_train, X_teste, y_test, y_test_alt, Xteste)



# Saving probabilities to send for testing
#train['prob_'] = prob_train.T
test['ATTRITION'] = prob_realtest

send = test[['ID_CORRELATIVO','ATTRITION']]

send.to_csv("output/testProb2_.csv", index=False)






