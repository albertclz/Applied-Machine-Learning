#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar  4 20:35:08 2018

@author: albertzhang
"""

import scipy.io
import numpy as np

#####load data
data = scipy.io.loadmat('/Users/albertzhang/Desktop/18spring/ML/HW/HW3/hw3data.mat')

def standardization(X):    
    X_scaled = np.zeros(X.shape)
    for i in range(X.shape[1]):
        mean = np.mean(X[:,i])
        std = np.std(X[:,i])
        for j in range(X.shape[0]):
            X_scaled[j,i] = (X[j,i]-mean)/std
    
    return X_scaled 

X = data['data']
X_scaled = standardization(X)   
y = np.int64(data['labels'])
y[y == 0] = -1


#### based on Problem4(b)          
n = len(y)
C = 10 / n 
alpha = np.zeros((n, 1))
Kernel = np.matmul(X_scaled, X_scaled.transpose())
for t in range(2):
    for i in range(n):
        a = 2 * y[i] * y[i] * Kernel[i, i]
        b = 1 - 2 * y[i] * (np.matmul(Kernel[i, :], np.multiply(alpha, y)) - Kernel[i, i] * alpha[i] * y[i])
        if C <= b / a:
            alpha[i] = C
        elif 0 >= b / a:
            alpha[i] = 0
        else:
            alpha[i] = b / a
value = np.sum(alpha) - sum(sum(np.multiply(np.matmul(np.multiply(y, alpha), np.multiply(y, alpha).transpose()),np.matmul(X_scaled, X_scaled.transpose()))))
weight = np.sum(np.multiply(np.multiply(y, alpha), X_scaled), axis = 0)


print("The objective value after 2 iterations is {}.".format(value))
print("The weight vector after two iterations is: {}".format(weight))





