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
import xgboost as xgb
import matplotlib.pyplot as plt

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

X_train, X_test, Xtest = X_train.T, X_test.T, Xtest.T
y_train, y_test = np.ravel(y_train.T), np.ravel(y_test.T)
#plt.matshow(pd.DataFrame(X_train).corr())

# Autoencoder implementation 


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
xg_cl.fit(X_train, y_train, early_stopping_rounds=40, eval_metric="logloss", eval_set=eval_set, verbose=True)

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