#!/usr/bin/env python
# coding: utf-8

"""
    Bbva Data Challenge
    ============================
    Machine learning model to predict the customer attrition probability
    
    > Imputation technique to deal with missing values

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
