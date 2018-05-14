#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 28 21:20:57 2018

@author: albertzhang
"""

import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_boston
boston_dataset = load_boston()
df = pd.DataFrame(boston_dataset['data'])

CRIM=list()
ZN=list()
INDUS=list()
CHAS=list()
NOX=list()
RM=list()
AGE=list()
DIS=list()
RAD=list()
TAX=list()
PTRATI0=list()
B=list()
LSTAT=list()


for i in range(len(boston_dataset['data'])):
    CRIM.append(boston_dataset['data'][i][0])
    ZN.append(boston_dataset['data'][i][1])
    INDUS.append(boston_dataset['data'][i][2])
    CHAS.append(boston_dataset['data'][i][3])
    NOX.append(boston_dataset['data'][i][4])
    RM.append(boston_dataset['data'][i][5])
    AGE.append(boston_dataset['data'][i][6])
    DIS.append(boston_dataset['data'][i][7])
    RAD.append(boston_dataset['data'][i][8])
    TAX.append(boston_dataset['data'][i][9])
    PTRATI0.append(boston_dataset['data'][i][10])
    B.append(boston_dataset['data'][i][11])
    LSTAT.append(boston_dataset['data'][i][12])
    
features=[CRIM,ZN,INDUS,CHAS,NOX,RM,AGE,DIS,RAD,TAX,PTRATI0,B,LSTAT]            


y = boston_dataset['target']

plt.subplots(1,13,figsize=(3,40))
plt.scatter(ZN,y)
plt.show()

fig, axes = plt.subplots(3,5,figsize=(15,10))

for i in range(0,3):
    for j in range(0,5):  
        if i*5+j<13:                  
            axes[i,j].scatter(features[i*5+j],y,alpha=0.2)
            axes[i,j].set_title(['CRIM','ZN','INDUS','CHAS','NOX','RM','AGE','DIS','RAD','TAX','PTRATI0','B','LSTAT'][i*5+j])
        else:
            axes[i,j].axis('off')
            axes[i,j].axis('off')
            
plt.suptitle('each feature against the MEDV')
plt.savefig('task33.png')
plt.show()



