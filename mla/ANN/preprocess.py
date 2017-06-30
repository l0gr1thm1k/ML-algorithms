# -*- coding: utf-8 -*-
"""
Created on Thu May 25 09:13:44 2017

@author: Daniel
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler

def preprocess(fname):
    """
    @desc   - preprocess a dataset into some training examples and
              target outputs
    """
    dataset = pd.read_csv(fname)
    
    for i in range(dataset.shape[1]):
        encoder = LabelEncoder()
        encoder.fit(dataset.iloc[:, i])
        dataset.iloc[:, i] = encoder.transform(dataset.iloc[:, i])
        
    X = dataset.iloc[:, :-1]
    y = dataset.iloc[:, -1]
    
    encoder = OneHotEncoder()
    X = encoder.fit_transform(X).toarray()
    
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    ones = np.ones(X.shape[0])
    ones = ones.reshape(-1, 1)
    X = np.append(ones, X, axis=1)
    return X, y