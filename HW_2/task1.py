#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  6 16:48:11 2018

@author: albertzhang
"""
import matplotlib.pyplot as plt
import numpy as np
import sklearn.datasets
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.model_selection import train_test_split, cross_val_score
############    1.1

california_dataset = sklearn.datasets.fetch_california_housing()

plt.plot(california_dataset['target'])
plt.show

fig, axes = plt.subplots(4,2,figsize=(15,10))
for i in range(0,4):
    for j in range(0,2):
        axes[i,j].plot(california_dataset['data'][:,i*2+j])
        axes[i,j].set_title(california_dataset['feature_names'][i*2+j])
plt.show()


############    1.2

fig, axes = plt.subplots(4,2,figsize=(15,10))
for i in range(0,4):
    for j in range(0,2):
        axes[i,j].scatter(california_dataset['data'][:,i*2+j],california_dataset['target'])
        axes[i,j].set_title(california_dataset['feature_names'][i*2+j])
plt.show()

############    1.3

X_train, X_test, y_train, y_test = train_test_split(california_dataset['data'],california_dataset['target'])
LRScore=np.mean(cross_val_score(LinearRegression(), X_train, y_train, cv=10))
RidgeScore=np.mean(cross_val_score(Ridge(), X_train, y_train, cv=10))
LassoScore=np.mean(cross_val_score(Lasso(), X_train, y_train, cv=10))
ElasticNetScore=np.mean(cross_val_score(ElasticNet(), X_train, y_train, cv=10))

print
