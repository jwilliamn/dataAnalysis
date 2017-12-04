#!/usr/bin/env python
# coding: utf-8

"""
    IBM attrition model
    ============================
    Machine learning models to predict the employee attrition using the
    IBM attrition dataset

    Structure:
        attrition.py

    _copyright_ = 'Copyright (c) 2017 J.W.'
    _license_ = GNU General Public License, see LICENSE for more details
"""

# Libraries ##
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.tools as tls

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss
from imblearn.over_sampling import SMOTE
import xgboost

from sklearn.preprocessing import LabelEncoder, OneHotEncoder

import warnings
warnings.filterwarnings('ignore')

from subprocess import check_output


# Reading data ##
attrition = pd.read_csv('WA_Fn-UseC_-HR-Employee-Attrition.csv')

# Data exploratory ##
#attrition.isnull().any()
#attrition.describe()
#attrition.std()

# Cleaning data
# EmployeeCount and StandardHours have std = 0, we have to drop them
attrition = attrition.drop('EmployeeCount', axis=1)
attrition = attrition.drop('StandardHours', axis=1)

# Dataset into features and labels
X = attrition.drop('Attrition', axis=1)
Y = attrition[['Attrition']]

# Encoding categorical data (strings into numbers)
labelencoderX = LabelEncoder()
X["BusinessTravel"] = labelencoderX.fit_transform(X["BusinessTravel"])
X["Department"] = labelencoderX.fit_transform(X["Department"])
X["EducationField"] = labelencoderX.fit_transform(X["EducationField"])
X["Gender"] = labelencoderX.fit_transform(X["Gender"])
X["JobRole"] = labelencoderX.fit_transform(X["JobRole"])
X["MaritalStatus"] = labelencoderX.fit_transform(X["MaritalStatus"])
X["Over18"] = labelencoderX.fit_transform(X["Over18"])
X["OverTime"] = labelencoderX.fit_transform(X["OverTime"])

labelencoderY= LabelEncoder()
Y = labelencoderY.fit_transform(Y)