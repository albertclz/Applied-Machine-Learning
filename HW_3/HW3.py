#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 28 19:43:18 2018

@author: albertzhang
"""

import numpy as np
import xlsxwriter
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split, cross_val_score 
from sklearn.preprocessing import StandardScaler, Imputer, PolynomialFeatures, LabelEncoder
from sklearn.feature_selection import RFECV, SelectKBest, SelectPercentile, SelectFpr, f_regression
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import make_pipeline


data2015 = pd.read_excel('/Users/albertzhang/Desktop/18spring/Applied_ML/HW/HW_3/2015 FE Guide-for DOE-Mobility Ventures only-OK to release-no-sales-4-27-2017Mercedesforpublic.xlsx')
data2016 = pd.read_excel('/Users/albertzhang/Desktop/18spring/Applied_ML/HW/HW_3/2016 FE Guide for DOE-OK to release-no-sales-4-27-2017Mercedesforpublic.xlsx')
data2017 = pd.read_excel('/Users/albertzhang/Desktop/18spring/Applied_ML/HW/HW_3/2017 FE Guide for DOE-release dates before 9-20-2017-no sales-9-19-2017MercedesCadillacforpublic.xlsx')
data2018 = pd.read_excel('/Users/albertzhang/Desktop/18spring/Applied_ML/HW/HW_3/2018 FE Guide for DOE-release dates before 2-24-2018-no-sales-2-23-2018public.xlsx')


trainData = data2015.append([data2016, data2017],ignore_index=True)

y_train = trainData['Comb Unrd Adj FE - Conventional Fuel'].to_frame()
X_train = trainData.drop(columns=['Comb Unrd Adj FE - Conventional Fuel'])
y_test = data2018['Comb Unrd Adj FE - Conventional Fuel'].to_frame()
X_test = data2018.drop(columns=['Comb Unrd Adj FE - Conventional Fuel'])

def ColumnsType(X_train):
    ### divide the columns into continuous and categorical
    CategoricalColumns = list()
    ContinuousColumns = list()
    for column in X_train.columns:
        if type(X_train[column].dropna().reset_index(drop=True)[0]) != np.int64 and type(X_train[column].dropna().reset_index(drop=True)[0]) != np.float64 and type(X_train[column].dropna().reset_index(drop=True)[0]) != int and type(X_train[column].dropna().reset_index(drop=True)[0]) != float:
            CategoricalColumns.append(column)
        else:
            ContinuousColumns.append(column)
    return ContinuousColumns,CategoricalColumns

def preprocessing(X_train):
    
    ### delete direct measurement columns
    keywords=['EPA','CO2','Smog','Guzzler','FE','MPG','Cost','Rating','Range']
    for word in keywords:
        for column in X_train.columns:
            if word in column:
                del X_train[column]
    print(X_train.shape)
                
    ### drop columns with the number of value more than 2/3 times of the whole dataset
    X_train = X_train.dropna(thresh=len(X_train)*(2/3), axis=1)
    ####### remove value 'Mod'
    
    
    modColumns=['MFR Calculated Gas Guzzler MPG ','FE Rating (1-10 rating on Label)','GHG Rating (1-10 rating on Label)','#1 Mfr Smog Rating (Mfr Smog 1-10 Rating on Label for Test Group 1)','City Unadj FE - Conventional Fuel','Hwy Unadj FE - Conventional Fuel','Comb Unadj FE - Conventional Fuel']
    for column in modColumns:
        try:
            X_train[column] = X_train[column].replace(to_replace='Mod',value=np.nan)
        except:
            continue
    print('strat to detect categorical columns')
    print(X_train.shape)

    return X_train


def imputScalOHE(X_train,X_test,ContinuousColumns,CategoricalColumns):
    '''
    OneHotEncoding on categorical columns;
    Imputation and Scaling on continuous columns
    '''    
    X_train_con = X_train[ContinuousColumns]
    X_train_cat = X_train[CategoricalColumns]
    del X_train_cat['Release Date']
    X_test_con = X_test[ContinuousColumns]
    X_test_cat = X_test[CategoricalColumns]
    del X_test_cat['Release Date']
    #####onehotencoding
    X_train_cat = pd.get_dummies(X_train_cat)
    X_test_cat = pd.get_dummies(X_test_cat)
    # Get missing columns in the training test
    missing_cols = set( X_train_cat.columns ) - set(X_test_cat.columns )
    # Add a missing column in test set with default value equal to 0
    for column in missing_cols:
        X_test_cat[column] = 0
    # Ensure the order of column in the test set is in the same order than in train set
    X_test_cat = X_test_cat[X_train_cat.columns]
    #####impute
    imputer = Imputer()
    imputer.fit(X_train_con)
    X_train_con_imputed = imputer.transform(X_train_con)
    X_test_con_imputed = imputer.transform(X_test_con)
    #####scaling
    scaler = StandardScaler()
    scaler.fit(X_train_con_imputed)    
    X_train_con_scaled_imputed = scaler.transform(X_train_con_imputed) 
    X_test_con_scaled_imputed = scaler.transform(X_test_con_imputed) 
    
    X_train_ISO =  np.concatenate((X_train_con_scaled_imputed,X_train_cat.as_matrix()),axis=1)
    X_test_ISO = np.concatenate((X_test_con_scaled_imputed,X_test_cat.as_matrix()),axis=1)
    
    ConColsNum, CatColsNum = X_train_con.shape[1], X_train_cat.shape[1]
    
    ##### ISO stands for Imputation Standardlization and Onehotencoding
    return X_train_ISO, X_test_ISO, ConColsNum, CatColsNum

def polyFeatures(X, ConColsNum, CatColsNum):
    X_con = X[:,:ConColsNum]
    X_cat = X[:,:(CatColsNum+ConColsNum)]
    
    poly = PolynomialFeatures()
    X_con_poly = poly.fit_transform(X_con)  
    scaler = StandardScaler()
    scaler.fit(X_con_poly)    
    X_con_poly_scaled = scaler.transform(X_con_poly)
    X_poly =  np.concatenate((X_con_poly_scaled,X_cat),axis=1)
    
    return X_poly

X_train_prep = preprocessing(X_train)
X_test_prep = preprocessing(X_test)
ContinuousColumns, CategoricalColumns = ColumnsType(X_train_prep)


X_train_ISO,X_test_ISO,ConColsNum,CatColsNum, ColNames = imputScalOHE(X_train_prep,X_test_prep,ContinuousColumns,CategoricalColumns)


LR=LinearRegression().fit(X_train_ISO,y_train)
RG=Ridge().fit(X_train_ISO,y_train)
LA=Lasso().fit(X_train_ISO,y_train)
EN=ElasticNet().fit(X_train_ISO,y_train)

print('LRScore:{}\nRidgeScore:{}\nLassoScore:{}\nElasticNetScore:{}'.format(LR.score(X_test_ISO,y_test),RG.score(X_test_ISO,y_test),LA.score(X_test_ISO,y_test),EN.score(X_test_ISO,y_test)))    
    


    
X_train_poly = polyFeatures(X_train_ISO,ConColsNum, CatColsNum)
X_test_poly = polyFeatures(X_test_ISO,ConColsNum, CatColsNum)

    
LR=LinearRegression().fit(X_train_poly,y_train)
RG=Ridge().fit(X_train_poly,y_train)
LA=Lasso().fit(X_train_poly,y_train)
EN=ElasticNet().fit(X_train_poly,y_train)

print('LRScore:{}\nRidgeScore:{}\nLassoScore:{}\nElasticNetScore:{}'.format(LR.score(X_test_poly,y_test),RG.score(X_test_poly,y_test),LA.score(X_test_poly,y_test),EN.score(X_test_poly,y_test)))    



GB = GradientBoostingRegressor().fit(X_train_ISO, y_train)
print("Gradient Boosting score: {}".format(GB.score(X_test_ISO, y_test)))

svr = SVR().fit(X_train_ISO, y_train)
print("SVM grid search score: {}".format(svr.score(X_test_ISO, y_test)))


select = SelectKBest(k=20, score_func=f_regression)
select.fit(X_train_ISO, y_train)
X_train_sub = select.transform(X_train_ISO)
X_test_sub = select.transform(X_test_ISO)
LR_selected = LinearRegression().fit(X_train_sub, y_train)
LR_selected.score(X_test_sub,y_test) 

    

important_features = []
for i in list(X_train):
    if abs(y_train.corr(X_train[i]))>0.03:
        important_features.append(i)
X_train=X_train[important_features]
X_test=X_test[important_features]


    
    
    