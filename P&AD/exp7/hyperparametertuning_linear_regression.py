# -*- coding: utf-8 -*-
"""
Created on Mon Sep 20 15:33:41 2021

@author: Dhruv Singhal
"""

#%%
import numpy as np
import matplotlib.pyplot  as plt
#import pandas as pd
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
#%%
from sklearn.datasets import make_regression
x,y=make_regression(n_samples=10000,n_features=5,noise=30)
sns.distplot(y)

#%%
plt.hist(x)
#%%
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,train_size=0.15,random_state=42)
from sklearn.metrics import r2_score,mean_squared_error
#%%
from sklearn.linear_model import LinearRegression
lin=LinearRegression()
lin.fit(x_train,y_train)
y_pred=lin.predict(x_test)
print("r2 score without tuning",r2_score(y_test,y_pred))
print("RMSE without tuning",np.sqrt(mean_squared_error(y_test,y_pred)))

#%%  Hyperparameter Tuning From Here
#%%
from sklearn.model_selection import GridSearchCV
tuned_parameters = [{'fit_intercept': ['True'], 'normalize': ['True']},
                    {'fit_intercept': ['False'], 'normalize': ['True']},
                    {'fit_intercept': ['True'], 'normalize': ['False']},
                    {'fit_intercept': ['False'], 'normalize': ['False']}
                    
                    ]
clf=GridSearchCV(LinearRegression(),tuned_parameters,scoring=('r2'))
clf.fit(x_train,y_train)
print("Best parameters set found on development set:")
print()
print(clf.best_params_)
print()
print("Best Score:",clf.best_score_)
z=clf.cv_results_

#%% kfold cross validation with hyperparameter tuning
from sklearn.model_selection import KFold
k = 5
kf = KFold(n_splits=k, random_state=None)
model = GridSearchCV(LinearRegression(),tuned_parameters,scoring=('r2'))
 
for train_index , test_index in kf.split(x):
    X_train , X_test = x[train_index,:],x[test_index,:]
    y_train , y_test = y[train_index] , y[test_index]
     
    model.fit(X_train,y_train)
    pred_values = model.predict(X_test)
print("Best parameters set found on development set:")
print()
print(model.best_params_)
print()
print("Best Score:",model.best_score_)
z2=model.cv_results_    
   
     
#%%




